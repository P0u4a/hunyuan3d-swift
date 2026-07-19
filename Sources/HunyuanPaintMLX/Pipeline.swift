import Foundation
import MLX
import MLXRandom

/// Weight loading from the original torch safetensors (NCHW conv → NHWC transpose + substring renames).
public enum Weights {
    /// Load a torch-layout checkpoint and keep floating-point weights compact. The published paint
    /// checkpoints contain a mixture of fp32/fp16/bf16 tensors; forcing every tensor to fp32 was
    /// the largest source of avoidable resident memory on Apple Silicon.
    public static func loadTorch(_ path: String, renames: [(String, String)] = [],
                                 dtype: DType = .float16) throws -> [String: MLXArray] {
        let sd = try loadArrays(url: URL(fileURLWithPath: path))
        var out = [String: MLXArray]()
        for (k0, v0) in sd {
            var k = k0
            for (a, b) in renames { k = k.replacingOccurrences(of: a, with: b) }
            let layout = v0.ndim == 4 ? v0.transposed(0, 2, 3, 1) : v0
            out[k] = layout.dtype.isFloatingPoint && layout.dtype != dtype ? layout.asType(dtype) : layout
        }
        return out
    }
    public static func splitPBR(_ all: [String: MLXArray]) -> (W, W) {
        var main = [String: MLXArray](), dual = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("unet_dual.") { dual[String(k.dropFirst(10))] = v }
            else if k.hasPrefix("unet.") { main[String(k.dropFirst(5))] = v }
        }
        return (W(main), W(dual))
    }
}

/// Result geometry + baked texture from a paint run. Format-agnostic — the caller
/// serializes to whatever mesh format it uses (Modelr writes a `.tmesh` + PNG).
public struct PaintResult {
    public let vertices: [Float]   // flat xyz, unwrapped geometry
    public let faces: [UInt32]     // flat triangle indices
    public let uvs: [Float]        // flat uv, viewer convention (v-flipped to top-left)
    public let albedoPNG: Data     // baked base-color texture as PNG bytes
}

/// PBR paint result: same geometry contract as `PaintResult`, plus the second baked map.
/// `metallicRoughnessPNG` uses the glTF channel packing the model produces: G = roughness,
/// B = metallic (R unused). Feed both PNGs to `writeGLB` or the app's own serializer.
public struct PBRPaintResult {
    public let vertices: [Float]            // flat xyz, unwrapped geometry
    public let faces: [UInt32]              // flat triangle indices
    public let uvs: [Float]                 // flat uv, viewer convention (v-flipped to top-left)
    public let albedoPNG: Data              // baked base-color texture as PNG bytes
    public let metallicRoughnessPNG: Data   // baked MR texture as PNG bytes (G=roughness, B=metallic)
}

/// Paint pipeline in Swift: mesh + image → textured geometry. Port of run_paint*.py.
/// Models are loaded by stage so the VAE, DINO, dual conditioner, main UNet and super-resolution
/// network do not all occupy unified memory at once. This trades some checkpoint I/O for a much
/// lower peak and makes a full PBR run practical on a 24 GB Mac.
public final class PaintPipeline {
    let weightsRoot: String
    public var res: Int, steps: Int, tex: Int    // per-run knobs; do not affect which weights load
    // Audit fix (b): CFG guidance is per-model (RGB/2.0 uses 2.0, PBR/2.1 uses 3.0 — matches
    // scripts/run_paint.py and scripts/run_paint_pbr.py). It was previously hardcoded to 3.0 for
    // both paths; it is now a per-method parameter with the correct default per model.
    let sf: Float = 0.18215
    let superRes: Bool
    let elevs: [Float] = [0, 0, 0, 0, 90, -90]
    let azims: [Float] = [0, 90, 180, 270, 0, 180]
    let vw: [Float] = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    public init(weightsRoot: String, res: Int = 512, steps: Int = 15, tex: Int = 4096,
                superRes: Bool = true, cacheLimitMB: Int = 128) {
        self.weightsRoot = weightsRoot; self.res = res; self.steps = steps; self.tex = tex; self.superRes = superRes
        MLX.Memory.cacheLimit = cacheLimitMB * 1024 * 1024
    }

    private func weightPath(_ candidates: String...) -> String {
        for relative in candidates {
            let path = "\(weightsRoot)/\(relative)"
            if FileManager.default.fileExists(atPath: path) { return path }
        }
        return "\(weightsRoot)/\(candidates[0])"   // preserve a useful load error with expected path
    }

    private func loadVAE() throws -> PaintVAE {
        PaintVAE(W(try Weights.loadTorch(
            weightPath("vae/diffusion_pytorch_model.safetensors",
                       "hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors"),
            renames: [(".to_out.0.", ".to_out.")])))
    }

    private func loadSR() -> RealESRGAN? {
        guard superRes,
              let arrs = try? loadArrays(url: URL(fileURLWithPath: weightPath(
                "realesrgan/rrdbnet_mlx.safetensors"))) else { return nil }
        return RealESRGAN(W(arrs.mapValues {
            $0.dtype.isFloatingPoint && $0.dtype != .float16 ? $0.asType(.float16) : $0
        }))
    }

    private func releaseStage() {
        MLX.Memory.clearCache()
    }

    private func memoryLine(_ stage: String) {
        func gib(_ n: Int) -> String { String(format: "%.2f", Double(n) / 1_073_741_824) }
        print("[memory] \(stage): active=\(gib(MLX.Memory.activeMemory)) GiB " +
              "cache=\(gib(MLX.Memory.cacheMemory)) GiB peak=\(gib(MLX.Memory.peakMemory)) GiB")
    }

    private func encodeControls(normals: [MLXArray], positions: [MLXArray],
                                imagePaths: [String])
        throws -> (normal: MLXArray, position: MLXArray, reference: MLXArray) {
        precondition(!imagePaths.isEmpty)
        let vae = try loadVAE()
        func enc(_ imgs: [MLXArray]) -> MLXArray {
            vae.encodeMean(stacked(imgs) * 2 - 1) * sf
        }
        let normal = enc(normals).expandedDimensions(axis: 0)
        eval(normal)
        let position = enc(positions).expandedDimensions(axis: 0)
        eval(position)
        // The checkpoint was trained with M reference images. Keep M as the second dimension;
        // the dual-stream UNet concatenates their per-layer tokens for reference attention.
        let reference = enc(imagePaths.map { prepRGB($0, res) }).expandedDimensions(axis: 0)
        eval(reference)
        return (normal, position, reference)
    }

    private func dinoHidden(_ imagePath: String) throws -> MLXArray {
        let dino = Dinov2(W(try Weights.loadTorch(weightPath(
            "dinov2/model.safetensors", "dinov2-giant/model.safetensors"))))
        let pixels = imagenetNorm(prepRGB(imagePath, 518)).expandedDimensions(axis: 0)
        let hidden = dino(pixels)
        eval(hidden)
        return hidden
    }

    private func preparedRGB(_ refLat: MLXArray)
        throws -> (main: W, ced: [String: MLXArray], gen: MLXArray) {
        let (main, dual) = Weights.splitPBR(try Weights.loadTorch(
            weightPath("hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors",
                       "unet/diffusion_pytorch_model.safetensors"),
            renames: [("transformer_blocks.0.transformer.", "transformer_blocks.0.")]))
        let ced = Paint20Wrapper(main: main, dual: dual).prepare(refLat: refLat)
        eval(Array(ced.values))
        let gen = main.a("learned_text_clip_gen")
        eval(gen)
        return (main, ced, gen)
    }

    private func preparedPBR(refLat: MLXArray, dinoHidden: MLXArray, posmap: MLXArray,
                             h: Int, n: Int)
        throws -> (main: W, ced: [String: MLXArray], dino: MLXArray,
                   rope: [Int: (MLXArray, MLXArray)]) {
        let (main, dual) = Weights.splitPBR(try Weights.loadTorch(
            weightPath("unet/diffusion_pytorch_model.safetensors",
                       "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors")))
        let prepared = PBRWrapper(main: main, dual: dual, nPbr: 2)
            .prepare(refLat: refLat, dinoHidden: dinoHidden, posmap: posmap, H: h, nGen: n)
        eval(Array(prepared.ced.values))
        eval(prepared.dino)
        let ropeArrays = prepared.rope.values.flatMap { [$0.0, $0.1] }
        if !ropeArrays.isEmpty { eval(ropeArrays) }
        return (main, prepared.ced, prepared.dino, prepared.rope)
    }

    private func denoiseRGB(normalLat: MLXArray, positionLat: MLXArray, refLat: MLXArray,
                            guidance: Float, seed: UInt64,
                            onProgress: ((String, Float) -> Void)?, isCancelled: () -> Bool,
                            onViews: ((Data) -> Void)?)
        throws -> MLXArray? {
        let prepared = try preparedRGB(refLat)
        releaseStage()                         // dual UNet is no longer referenced after prepare
        memoryLine("RGB conditioner released")
        let wrap = Paint20Wrapper(main: prepared.main, dual: W([:]))
        let (sig, ts) = uniPCSchedule(steps)
        let sched = UniPCScheduler(sigmas: sig, timesteps: ts)
        MLXRandom.seed(seed)
        let n = elevs.count, h = res / 8
        var latents = MLXRandom.normal([1, n, h, h, 4])
        let neg = zeros(prepared.gen.shape)
        let camGen = (0..<n).map { Int32($0) }
        for (i, t) in ts.enumerated() {
            if isCancelled() { return nil }
            let tArr = MLXArray(Array(repeating: Float(t), count: n))
            let vc = wrap.predict(latents, tArr, text: prepared.gen, normalLat: normalLat,
                                  positionLat: positionLat, camGen: camGen, ced: prepared.ced,
                                  mvaScale: 1, refScale: 1)
            eval(vc)                            // do not retain two complete lazy UNet graphs
            let vu = wrap.predict(latents, tArr, text: neg, normalLat: normalLat,
                                  positionLat: positionLat, camGen: camGen, ced: nil,
                                  mvaScale: 1, refScale: 0)
            eval(vu)
            latents = sched.step(vu + guidance * (vc - vu), t, latents)
            eval(latents)
            onProgress?("Painting (\(i+1)/\(steps))", 0.15 + 0.6 * Float(i + 1) / Float(steps))
            releaseStage()
            if let onViews, i % 3 == 2 || i == steps - 1,
               let data = try previewPNG(latents[0]) {
                onViews(data)
                releaseStage()
            }
        }
        return latents
    }

    private func denoisePBR(normalLat: MLXArray, positionLat: MLXArray, refLat: MLXArray,
                            dinoHidden: MLXArray, posmap: MLXArray, guidance: Float, seed: UInt64,
                            onProgress: ((String, Float) -> Void)?, isCancelled: () -> Bool,
                            onViews: ((Data) -> Void)?)
        throws -> MLXArray? {
        let n = elevs.count, h = res / 8
        let prepared = try preparedPBR(refLat: refLat, dinoHidden: dinoHidden, posmap: posmap,
                                       h: h, n: n)
        releaseStage()                         // dual UNet graph + weights are now reclaimable
        memoryLine("PBR conditioner released")
        let wrap = PBRWrapper(main: prepared.main, dual: W([:]), nPbr: 2)
        let (sig, ts) = uniPCSchedule(steps)
        let sched = UniPCScheduler(sigmas: sig, timesteps: ts)
        MLXRandom.seed(seed)
        var latents = MLXRandom.normal([1, 2, n, h, h, 4])
        let dinoZero = zeros(prepared.dino.shape)
        let nb = 2 * n
        for (i, t) in ts.enumerated() {
            if isCancelled() { return nil }
            let tArr = MLXArray(Array(repeating: Float(t), count: nb))
            let vc = wrap.predict(latents, tArr, normalLat: normalLat, positionLat: positionLat,
                                  ced: prepared.ced, dino: prepared.dino, rope: prepared.rope,
                                  mvaScale: 1, refScale: 1)
            eval(vc)                            // serialize CFG to halve peak activation graphs
            let vu = wrap.predict(latents, tArr, normalLat: normalLat, positionLat: positionLat,
                                  ced: nil, dino: dinoZero, rope: prepared.rope,
                                  mvaScale: 1, refScale: 0)
            eval(vu)
            latents = sched.step(vu + guidance * (vc - vu), t, latents)
            eval(latents)
            onProgress?("Painting (\(i+1)/\(steps))", 0.15 + 0.6 * Float(i + 1) / Float(steps))
            releaseStage()
            if let onViews, i % 3 == 2 || i == steps - 1,
               let data = try previewPNG(latents[0, 0]) {
                onViews(data)
                releaseStage()
            }
        }
        return latents
    }

    private func decodeViews(_ latents: [MLXArray]) throws -> [[MLXArray]] {
        let vae = try loadVAE()
        var result = [[MLXArray]]()
        for latent in latents {
            let decoded = clip((vae.decode(latent / sf) + 1) / 2, min: 0, max: 1)
            eval(decoded)
            result.append((0..<elevs.count).map { decoded[$0] })
        }
        eval(result.flatMap { $0 })
        return result
    }

    private func previewPNG(_ latent: MLXArray) throws -> Data? {
        let views = try decodeViews([latent])[0]
        return pngData(concatenated(views, axis: 1))
    }

    private func superResolve(_ groups: [[MLXArray]]) -> [[MLXArray]] {
        guard let sr = loadSR() else { return groups }
        var result = [[MLXArray]]()
        for group in groups {
            var upscaled = [MLXArray]()
            for view in group {
                let up = clip(sr(view.expandedDimensions(axis: 0))[0], min: 0, max: 1)
                eval(up)                        // one view at a time; never retain 12 SR graphs
                upscaled.append(up)
                releaseStage()
            }
            result.append(upscaled)
        }
        return result
    }

    /// CLI-shaped PBR paint: file paths in, GLB out. Thin shell over `paintPBR` — the pipeline
    /// core is shared with the app entry point; this only loads the mesh, writes the debug
    /// texture PNGs next to the output, and serializes the GLB.
    public func run(meshPath: String, imagePath: String, outGLB: String,
                    referenceImagePaths: [String] = [],
                    guidance: Float = 3.0, seed: UInt64 = 0) throws {
        let t0 = Date()
        func log(_ s: String) { print("[pipeline] \(s)  (\(Int(-t0.timeIntervalSinceNow))s)") }
        let mesh = loadMesh(meshPath)
        guard let r = try paintPBR(mesh: mesh, imagePath: imagePath,
                                   referenceImagePaths: referenceImagePaths,
                                   guidance: guidance, seed: seed,
                                   debugPathPrefix: outGLB,
                                   onProgress: { s, _ in log(s) }) else { return }
        // debug: the baked textures next to the GLB (same bytes that get embedded)
        try r.albedoPNG.write(to: URL(fileURLWithPath: "\(outGLB).albedo.png"))
        try r.metallicRoughnessPNG.write(to: URL(fileURLWithPath: "\(outGLB).mr.png"))
        try writeGLB(path: outGLB, vertices: r.vertices, faces: r.faces, uvs: r.uvs,
                     baseColorPNG: r.albedoPNG, metallicRoughnessPNG: r.metallicRoughnessPNG)
        log("DONE → \(outGLB)")
    }

    /// 2.0 RGB paint: geometry + reference image → unwrapped geometry + baked base-color texture.
    /// Polls `isCancelled` (returns nil if it fires); streams decoded view grids via `onViews`.
    public func paintRGB(mesh: LoadedMesh, imagePath: String,
                         referenceImagePaths: [String] = [], guidance: Float = 2.0,
                         seed: UInt64 = 0,
                         onProgress: ((String, Float) -> Void)? = nil,
                         isCancelled: () -> Bool = { false },
                         onViews: ((Data) -> Void)? = nil) throws -> PaintResult? {
        onProgress?("Unwrapping UVs", 0.05)
        guard let uw = xatlasUnwrap(vertices: mesh.vertices, vertexCount: mesh.vertexCount,
                                    faces: mesh.faces, faceCount: mesh.faceCount) else { return nil }
        var V = [Float](repeating: 0, count: uw.vertexCount * 3)
        for i in 0..<uw.vertexCount { let o = Int(uw.vmapping[i]) * 3; V[i*3] = mesh.vertices[o]; V[i*3+1] = mesh.vertices[o+1]; V[i*3+2] = mesh.vertices[o+2] }
        let R = MeshRender(); R.loadMesh(V, uw.indices); R.setUV(uw.uvs, flipV: true)
        if isCancelled() { return nil }

        onProgress?("Rendering control maps", 0.1)
        let ctrl = zip(elevs, azims).map { R.renderControl($0.0, $0.1, res) }
        let normals = ctrl.map { $0.0 }, positions = ctrl.map { $0.1 }
        let allReferences = [imagePath] + referenceImagePaths
        onProgress?("Encoding controls + \(allReferences.count) reference(s)", 0.12)
        let encoded = try encodeControls(normals: normals, positions: positions,
                                         imagePaths: allReferences)
        releaseStage()
        memoryLine("VAE encoder released")
        if isCancelled() { return nil }

        guard let latents = try denoiseRGB(normalLat: encoded.normal, positionLat: encoded.position,
                                           refLat: encoded.reference, guidance: guidance, seed: seed,
                                           onProgress: onProgress, isCancelled: isCancelled,
                                           onViews: onViews) else { return nil }
        releaseStage()
        memoryLine("RGB denoiser released")
        if isCancelled() { return nil }

        onProgress?("Decoding views", 0.8)
        var views = try decodeViews([latents[0]])[0]
        releaseStage()
        memoryLine("VAE decoder released")
        if superRes {
            onProgress?("Super-resolving", 0.88)
            views = superResolve([views])[0]
            releaseStage()
            memoryLine("super-resolution released")
        }
        if isCancelled() { return nil }

        onProgress?("Baking texture", 0.93)
        let (texs, covered) = R.bakeMulti([views], elevs, azims, textureSize: tex, weights: vw)
        let texC = MeshRender.inpaint(texs[0], covered); eval(texC)
        guard let albedoPNG = pngData(texC) else { return nil }
        var uvOut = uw.uvs
        for i in 0..<(uvOut.count / 2) { uvOut[i*2+1] = 1 - uvOut[i*2+1] }          // v-flip → viewer top-left
        onProgress?("Done", 1.0)
        return PaintResult(vertices: V, faces: uw.indices, uvs: uvOut, albedoPNG: albedoPNG)
    }

    /// 2.1 PBR paint: geometry + reference image → unwrapped geometry + baked albedo and
    /// metallic-roughness textures. Same contract as `paintRGB`: polls `isCancelled` (returns
    /// nil if it fires), streams decoded albedo view grids via `onViews`, reports stages via
    /// `onProgress`. Debug artifacts are written only when `debugPathPrefix` is set (the CLI
    /// passes the output GLB path): `<prefix>.views.png` and `<prefix>.rendercheck.png`.
    public func paintPBR(mesh: LoadedMesh, imagePath: String,
                         referenceImagePaths: [String] = [], guidance: Float = 3.0,
                         seed: UInt64 = 0,
                         debugPathPrefix: String? = nil,
                         onProgress: ((String, Float) -> Void)? = nil,
                         isCancelled: () -> Bool = { false },
                         onViews: ((Data) -> Void)? = nil) throws -> PBRPaintResult? {
        onProgress?("Unwrapping UVs", 0.05)
        guard let uw = xatlasUnwrap(vertices: mesh.vertices, vertexCount: mesh.vertexCount,
                                    faces: mesh.faces, faceCount: mesh.faceCount) else { return nil }
        var V = [Float](repeating: 0, count: uw.vertexCount * 3)               // original geometry gathered by vmapping
        for i in 0..<uw.vertexCount { let o = Int(uw.vmapping[i]) * 3; V[i*3] = mesh.vertices[o]; V[i*3+1] = mesh.vertices[o+1]; V[i*3+2] = mesh.vertices[o+2] }
        let R = MeshRender(); R.loadMesh(V, uw.indices); R.setUV(uw.uvs, flipV: true)
        if isCancelled() { return nil }

        onProgress?("Rendering control maps", 0.1)
        let ctrl = zip(elevs, azims).map { R.renderControl($0.0, $0.1, res) }
        let normals = ctrl.map { $0.0 }, positions = ctrl.map { $0.1 }
        let allReferences = [imagePath] + referenceImagePaths
        onProgress?("Encoding controls + \(allReferences.count) reference(s)", 0.12)
        let encoded = try encodeControls(normals: normals, positions: positions,
                                         imagePaths: allReferences)
        releaseStage()
        memoryLine("VAE encoder released")
        onProgress?("Encoding reference", 0.14)
        // The original 2.1 model intentionally feeds only cond_imgs[:, :1] to DINO. Additional
        // images enrich dual-stream reference attention without changing global identity/layout.
        let dinoHS = try dinoHidden(imagePath)
        releaseStage()
        memoryLine("DINO released")
        let posmap = stacked(positions).expandedDimensions(axis: 0)            // [1,N,res,res,3]
        eval(posmap)
        if isCancelled() { return nil }

        guard let latents = try denoisePBR(normalLat: encoded.normal, positionLat: encoded.position,
                                           refLat: encoded.reference, dinoHidden: dinoHS,
                                           posmap: posmap, guidance: guidance, seed: seed,
                                           onProgress: onProgress, isCancelled: isCancelled,
                                           onViews: onViews) else { return nil }
        releaseStage()
        memoryLine("PBR denoiser released")
        if isCancelled() { return nil }

        onProgress?("Decoding views", 0.8)
        var decoded = try decodeViews([latents[0, 0], latents[0, 1]])
        var alb = decoded[0], mr = decoded[1]
        decoded.removeAll(keepingCapacity: false)
        releaseStage()
        memoryLine("VAE decoder released")
        if let p = debugPathPrefix { saveRGB(concatenated(alb, axis: 1), "\(p).views.png") }  // debug: albedo views grid
        if superRes {
            onProgress?("Super-resolving", 0.88)
            let upscaled = superResolve([alb, mr])
            alb = upscaled[0]; mr = upscaled[1]
            releaseStage()
            memoryLine("super-resolution released")
        }
        if isCancelled() { return nil }

        onProgress?("Baking textures", 0.93)
        let (texs, covered) = R.bakeMulti([alb, mr], elevs, azims, textureSize: tex, weights: vw)
        let texA = MeshRender.inpaint(texs[0], covered), texM = MeshRender.inpaint(texs[1], covered)
        eval(texA, texM)
        if let p = debugPathPrefix {
            // debug: render the texture back onto the mesh (bypasses GLB) at 3 angles
            let dbg = [R.renderTextured(0, 20, 420, texA), R.renderTextured(0, 140, 420, texA), R.renderTextured(0, 260, 420, texA)]
            saveRGB(concatenated(dbg, axis: 1), "\(p).rendercheck.png")
        }
        guard let albedoPNG = pngData(texA), let mrPNG = pngData(texM) else { return nil }
        var uvOut = uw.uvs
        for i in 0..<(uvOut.count / 2) { uvOut[i*2+1] = 1 - uvOut[i*2+1] }          // v-flip → viewer top-left
        onProgress?("Done", 1.0)
        return PBRPaintResult(vertices: V, faces: uw.indices, uvs: uvOut,
                              albedoPNG: albedoPNG, metallicRoughnessPNG: mrPNG)
    }
}

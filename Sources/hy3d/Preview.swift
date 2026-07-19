import Foundation
import MLX
import HunyuanPaintMLX

// MARK: - hy3d preview (geometry-only normal render)

func cmdPreview(_ args: Args) throws {
    guard let meshPath = args.positional.first else {
        throw CLIError("preview: missing <mesh.glb|obj>")
    }
    guard let out = args.str("o", "output") else {
        throw CLIError("preview: missing -o <out.png>")
    }
    let res = args.int("res") ?? 420
    let ssaa = args.int("ssaa") ?? 2
    guard res > 0 else { throw CLIError("preview: --res must be positive") }
    guard (1...4).contains(ssaa) else { throw CLIError("preview: --ssaa must be between 1 and 4") }

    let mesh = loadMesh(meshPath)
    guard mesh.vertexCount > 0 else { throw CLIError("preview: failed to load mesh \(meshPath)") }
    let renderer = MeshRender()
    renderer.loadMesh(mesh.vertices, mesh.faces)
    let angles: [Float] = [20, 140, 260]
    let views = angles.map { angle -> MLXArray in
        let (normal, _) = renderer.renderControl(0, angle, res * ssaa)
        return MeshRender.downsampleSSAA(normal, scale: ssaa)
    }
    let strip = concatenated(views, axis: 1)
    eval(strip)
    saveRGB(strip, out)
    print("preview: wrote \(out) (\(res)px x 3, \(ssaa)x SSAA)")
}

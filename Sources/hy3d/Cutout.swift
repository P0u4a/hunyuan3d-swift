import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

// MARK: - hy3d cutout

/// Use the native macOS foreground-instance model to produce the segmented RGBA input expected by
/// Hunyuan's shape preprocessor. Keeping this explicit avoids silently changing already-matted art.
func cmdCutout(_ args: Args) throws {
    guard args.positional.count == 2 else {
        throw CLIError("cutout: usage: hy3d cutout <input.png> <output.png>")
    }
    guard #available(macOS 14.0, *) else {
        throw CLIError("cutout: foreground instance masks require macOS 14 or newer")
    }
    let input = args.positional[0], output = args.positional[1]
    guard let image = loadCGImage(input) else { throw CLIError("cutout: cannot read image \(input)") }

    let request = VNGenerateForegroundInstanceMaskRequest()
    let handler = VNImageRequestHandler(cgImage: image, orientation: .up)
    try handler.perform([request])
    guard let observation = request.results?.first,
          !observation.allInstances.isEmpty else {
        throw CLIError("cutout: no foreground subject found")
    }

    let mask = try observation.generateScaledMaskForImage(
        forInstances: observation.allInstances,
        from: handler
    )
    let cutout = try applyForegroundMask(mask, to: image)
    try writeCutoutPNG(cutout, to: output)
    print("cutout: wrote \(output) (\(cutout.width)x\(cutout.height), RGBA)")
}

private func applyForegroundMask(_ mask: CVPixelBuffer, to image: CGImage) throws -> CGImage {
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
    guard let maskBase = CVPixelBufferGetBaseAddress(mask) else {
        throw CLIError("cutout: could not read foreground mask")
    }

    let width = image.width, height = image.height, bytesPerRow = width * 4
    guard CVPixelBufferGetWidth(mask) == width, CVPixelBufferGetHeight(mask) == height else {
        throw CLIError("cutout: foreground mask dimensions do not match the image")
    }
    let format = CVPixelBufferGetPixelFormatType(mask)
    guard format == kCVPixelFormatType_OneComponent8
            || format == kCVPixelFormatType_OneComponent32Float else {
        throw CLIError("cutout: unsupported foreground mask pixel format \(format)")
    }

    var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
    guard let sourceContext = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { throw CLIError("cutout: could not allocate image buffer") }
    sourceContext.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    let maskStride = CVPixelBufferGetBytesPerRow(mask)
    for y in 0..<height {
        for x in 0..<width {
            let amount: UInt16
            if format == kCVPixelFormatType_OneComponent8 {
                let row = maskBase.advanced(by: y * maskStride).assumingMemoryBound(to: UInt8.self)
                amount = UInt16(row[x])
            } else {
                let row = maskBase.advanced(by: y * maskStride).assumingMemoryBound(to: Float.self)
                amount = UInt16((min(max(row[x], 0), 1) * 255).rounded())
            }
            let p = y * bytesPerRow + x * 4
            // The buffer is premultiplied RGBA, so scale color and alpha together at soft edges.
            for channel in 0..<4 {
                rgba[p + channel] = UInt8((UInt16(rgba[p + channel]) * amount + 127) / 255)
            }
        }
    }

    guard let outputContext = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let output = outputContext.makeImage() else {
        throw CLIError("cutout: could not create output image")
    }
    return output
}

private func writeCutoutPNG(_ image: CGImage, to path: String) throws {
    guard let destination = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else { throw CLIError("cutout: cannot create output \(path)") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw CLIError("cutout: failed to write \(path)")
    }
}

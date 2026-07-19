# hunyuan3d-swift
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/534826fa-0a79-45f0-a5af-8c69a49e1fe9" />

swift and python mlx ports of the hunyuan3d shape and paint pipelines.

the repo has two parts:

- swift package at the root
- python-mlx ports under `python/`

the swift code is checked against python fixtures. the python ports are checked against the original pytorch code.

## this is exciting
| run | config | wall time | peak memory |
|---|---|---:|---:|
| `hy3d shape` (small) | 30-step cfg, octree 256 | 20.9 s | ~5.6 gb |
| `hy3d shape` (large) | 8-step turbo, octree 256 | 22.3 s | ~7.3 gb |
| `hy3d paint` (rgb) | 512 render, 15 steps, +sr | 231 s | ~38 gb |
| `hy3d paint` (pbr) | 512 render, 15 steps, 4096 atlas, +sr | 344 s | ~39 gb |
| `hy3d generate` (small / rgb) | chained | 240 s | ~25 gb |
| `hy3d generate` (large / pbr) | chained | 360 s | ~33 gb |

the low-memory pipeline on `codex/21gb-low-memory` has also been measured end-to-end on an
m4 pro with 24 gb unified memory. small shape + 2.1 pbr paint at 512, 15 steps, a 4096 atlas,
and x4 super-resolution completed in 420 s with a 17.50 gib mlx peak and zero swaps. models are
loaded by stage, both cfg passes are evaluated sequentially, dino is evaluated one block at a
time, and the mlx cache defaults to 128 mb.

that's hunyuan3D-shape running in FIVE POINT SIX gigabytes of RAM (thanks to MLX). with a Q8 or Q4, we are easily in mobile territory. now, what would you do with a image to 3d model running on your phone (iPhone 15 onwards)? not sure, but that's really really cool, the first time it has been possible (AFAIK). here's some demos (checkout [Modelr](https://github.com/ZimengXiong/Modelr):

https://github.com/user-attachments/assets/495de7e8-6c76-4b3f-af37-ffa23e0b0a64

https://github.com/user-attachments/assets/7c1f5008-8a68-4e08-b3f6-80864d3c1a00


## package

main products:

- Hy3DMLX for shape generation
- HunyuanPaintMLX for texture generation
- `hy3d` for the command line

main commands:

```bash
swift build -c release
swift run -c release hy3d --help
```

`swift build` is useful as a compile check, but swiftpm cannot build mlx's metal shader library.
for a runnable command-line binary, use xcode's package builder:

```bash
xcodebuild -scheme hy3d -destination 'platform=macOS' -configuration Release \
  -derivedDataPath .build/xcode CODE_SIGNING_ALLOWED=NO build
.build/xcode/Build/Products/Release/hy3d --help
```

## weights

download model weights into local folders:

```bash
hf download zimengxiong/hunyuan3d-mlx-shape-small --local-dir weights/shape-small
hf download zimengxiong/hunyuan3d-mlx-paint-large --local-dir weights/paint-large
```

four supported slots:

- shape small: `hunyuan3d-dit-v2-mini`
- shape large: `hunyuan3d-dit-v2-0-turbo`
- paint small: `hunyuan3d-paint-v2-0`
- paint large: `hunyuan3d-paintpbr-v2-1`

## run

shape plus paint:

```bash
swift run -c release hy3d generate photo.png -o out.glb \
  --shape-weights weights/shape-small \
  --paint-weights weights/paint-large \
  --paint-model pbr --cache-mb 128
```

shape only:

```bash
swift run -c release hy3d shape photo.png -o mesh.glb \
  --weights weights/shape-small
```

paint an existing mesh:

```bash
swift run -c release hy3d paint mesh.glb photo.png -o textured.glb \
  --weights weights/paint-large --model pbr
```

additional paint references (repeat the option up to four times for detail crops or coherent alternate views):

```bash
.build/xcode/Build/Products/Release/hy3d paint mesh.glb primary.png \
  -o textured.glb --weights weights/paint-large --model pbr \
  --paint-ref face-and-shirt-detail.png \
  --paint-ref rear-or-three-quarter-view.png
```

the primary image remains the global dino reference. all images, including repeated
`--paint-ref` values, are VAE-encoded as separate reference-attention inputs. this improves
texture and visible detail conditioning; it does not add geometric detail to the shape mesh.

shape generation expects the subject on a transparent background. for opaque photos or renders,
create a native macOS Vision cutout first (macOS 14+):

```bash
.build/xcode/Build/Products/Release/hy3d cutout source.png source-cutout.png
.build/xcode/Build/Products/Release/hy3d generate source-cutout.png \
  -o model.glb --shape-weights weights/shape-small --paint-weights weights/paint-large
```

without this step, gradients or studio backdrops can be reconstructed as unwanted planar geometry.

replace `swift run -c release hy3d` in the examples above with
`.build/xcode/Build/Products/Release/hy3d` for the metal-enabled command-line build.

## license

swift source is mit. dependencies, model weights, and algorithm ports keep their own licenses.
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

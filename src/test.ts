import { Particle } from "./particle";
import { WebGPURenderer } from "./WebGPURenderer";

// export const Initialize = async () => {
//   // Canvas and WebGPU Initialization
//   const canvas: HTMLCanvasElement = <HTMLCanvasElement>(
//     document.getElementById("gfx-main")
//   );
//   const adapter: GPUAdapter = <GPUAdapter>(
//     await navigator.gpu?.requestAdapter()
//   );
//   const device: GPUDevice = <GPUDevice>await adapter?.requestDevice();
//   const context: GPUCanvasContext = <GPUCanvasContext>(
//     canvas.getContext("webgpu")
//   );
//   const format: GPUTextureFormat = "bgra8unorm";
//   context.configure({
//     device: device,
//     format: format,
//     alphaMode: "opaque",
//   });

//   // Shader Module
//   const shaders = `struct VertexInput {
//     @location(0) position : vec2<f32>,
//     @location(1) color : vec3<f32>,
//   };

//   struct Fragment {
//     @builtin(position) Position : vec4<f32>,
//     @location(0) Color : vec4<f32>,
//   };

//   @vertex
//   fn vs_main(vertexInput: VertexInput) -> Fragment {
//     var output : Fragment;
//     output.Position = vec4<f32>(vertexInput.position, 0.0, 1.0);
//     output.Color = vec4<f32>(vertexInput.color, 1.0);
//     return output;
//   }

//   @fragment
//   fn fs_main(@location(0) Color: vec4<f32>) -> @location(0) vec4<f32> {
//     return Color;
//   }`;
//   const shaderModule = device.createShaderModule({ code: shaders });

//   // Particle Setup
//   const particles = [
//     new Particle([0.0, 0.5], [1.0, 1.0, 0.0]),
//     new Particle([-0.5, -0.5], [0.0, 1.0, 1.0]),
//     new Particle([0.5, -0.5], [1.0, 0.0, 1.0])
//   ];

//   // Extract positions and colors from particles
//   const positions = particles.map(p => p.position);
//   const colors = particles.map(p => p.color);

//   // Create Vertex Buffers
//   const positionBuffer = device.createBuffer({
//     size: positions.length * 8, // Assuming vec2<f32> is 8 bytes
//     usage: GPUBufferUsage.VERTEX,
//     mappedAtCreation: true,
//   });
//   new Float32Array(positionBuffer.getMappedRange()).set(positions.flat());
//   positionBuffer.unmap();

//   const colorBuffer = device.createBuffer({
//     size: colors.length * 12, // Assuming vec3<f32> is 12 bytes
//     usage: GPUBufferUsage.VERTEX,
//     mappedAtCreation: true,
//   });
//   new Float32Array(colorBuffer.getMappedRange()).set(colors.flat());
//   colorBuffer.unmap();

//   // Pipeline Setup
//   const bindGroupLayout = device.createBindGroupLayout({ entries: [] });
//   const bindGroup = device.createBindGroup({
//     layout: bindGroupLayout,
//     entries: [],
//   });
//   const pipelineLayout = device.createPipelineLayout({
//     bindGroupLayouts: [bindGroupLayout],
//   });
//   const pipeline = device.createRenderPipeline({
//     layout: pipelineLayout,
//     vertex: {
//       module: shaderModule,
//       entryPoint: "vs_main",
//       buffers: [
//         {
//           arrayStride: 8,
//           attributes: [{ // Position
//             shaderLocation: 0,
//             offset: 0,
//             format: "float32x2"
//           }]
//         },
//         {
//           arrayStride: 12,
//           attributes: [{ // Color
//             shaderLocation: 1,
//             offset: 0,
//             format: "float32x3"
//           }]
//         }
//       ]
//     },
//     fragment: {
//       module: shaderModule,
//       entryPoint: "fs_main",
//       targets: [{ format: format }],
//     },
//     primitive: {
//       topology: "triangle-list",
//     },
//   });

//   // Render Pass
//   const commandEncoder = device.createCommandEncoder();
//   const textureView = context.getCurrentTexture().createView();
//   const renderpass = commandEncoder.beginRenderPass({
//     colorAttachments: [{
//       view: textureView,
//       loadOp: "clear",
//       storeOp: "store",
//       clearValue: { r: 0.25, g: 1.0, b: 0.5, a: 1.0 }, // Gray background
//     }],
//   });

//   renderpass.setPipeline(pipeline);
//   renderpass.setVertexBuffer(0, positionBuffer);
//   renderpass.setVertexBuffer(1, colorBuffer);
//   renderpass.setBindGroup(0, bindGroup);
//   renderpass.draw(3, 1, 0, 0);
//   renderpass.end();

//   // Submitting the Command
//   const queue = device.queue;
//   queue.submit([commandEncoder.finish()]);
// };


export const Initialize = async () => {
  
  const sceneManager = new WebGPURenderer("gfx-main");
  sceneManager.init().then(() => {
    sceneManager.createParticles(100); // Create 100 particles
    sceneManager.createBuffers();
    sceneManager.createPipeline();
    sceneManager.render();
  });
}
import { Particle } from "./particle";

export class WebGPURenderer {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice; // Marked as optional with "!"
  private context!: GPUCanvasContext; // Marked as optional with "!"
  private format: GPUTextureFormat = "bgra8unorm";
  private particles: Particle[];
  private pipeline!: GPURenderPipeline; // Marked as optional with "!"
  private positionBuffer!: GPUBuffer; // Marked as optional with "!"
  private colorBuffer!: GPUBuffer; // Marked as optional with "!"
  private queue!: GPUQueue; // Marked as optional with "!"

  constructor(canvasId: string) {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    this.particles = [];
  }

  async init() {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      throw new Error("Failed to get GPU adapter");
    }
    this.device = await adapter?.requestDevice();
    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });
    this.queue = this.device.queue;
  }

  createParticles(numParticles: number) {
    for (let i = 0; i < numParticles; i++) {
      // Explicitly declare the position as a tuple [number, number]
      const position: [number, number] = [
        Math.random() * 2 - 1, // Random position in range [-1, 1]
        Math.random() * 2 - 1,
      ];
      const color: [number, number, number] = [
        Math.random(), // Random color
        Math.random(),
        Math.random(),
      ];
      this.particles.push(new Particle(position, color));
    }
  }

  createBuffers() {
    const positions = this.particles.map(p => p.position).flat();
    const colors = this.particles.map(p => p.color).flat();

    this.positionBuffer = this.device.createBuffer({
      size: positions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
    this.positionBuffer.unmap();

    this.colorBuffer = this.device.createBuffer({
      size: colors.length * 12,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.colorBuffer.getMappedRange()).set(colors);
    this.colorBuffer.unmap();
  }

  createPipeline() {
    // Shader Module
    const shaders = `struct VertexInput {
      @location(0) position : vec2<f32>,
      @location(1) color : vec3<f32>,
    };
  
    struct Fragment {
      @builtin(position) Position : vec4<f32>,
      @location(0) Color : vec4<f32>,
    };
  
    @vertex
    fn vs_main(vertexInput: VertexInput) -> Fragment {
      var output : Fragment;
      output.Position = vec4<f32>(vertexInput.position, 0.0, 1.0);
      output.Color = vec4<f32>(vertexInput.color, 1.0);
      return output;
    }
  
    @fragment
    fn fs_main(@location(0) Color: vec4<f32>) -> @location(0) vec4<f32> {
      return Color;
    }`;
  
    const shaderModule = this.device.createShaderModule({ code: shaders });
  
    // Bind Group Layout (if any resources need to be bound to the pipeline)
    const bindGroupLayout = this.device.createBindGroupLayout({ entries: [] });
  
    // Pipeline Layout
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
  
    // Render Pipeline
    this.pipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 8, // size of each position vertex (vec2<f32>)
            attributes: [{ // Position attribute
              shaderLocation: 0,
              offset: 0,
              format: "float32x2"
            }]
          },
          {
            arrayStride: 12, // size of each color vertex (vec3<f32>)
            attributes: [{ // Color attribute
              shaderLocation: 1,
              offset: 0,
              format: "float32x3"
            }]
          }
        ]
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: "point-list", // or "point-list" if you are drawing points
      },
    });
  }
  

  render() {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
  
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        loadOp: 'clear',
        clearValue: { r: 0.25, g: 0.25, b: 0.25, a: 1.0 },
        storeOp: 'store',
      }],
    };
  
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setVertexBuffer(1, this.colorBuffer);
    
    // If you decide to keep bind group layout, create and set an empty bind group here
    const emptyBindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [] });
    passEncoder.setBindGroup(0, emptyBindGroup);
  
    passEncoder.draw(this.particles.length, 1, 0, 0);
    passEncoder.end();
  
    this.queue.submit([commandEncoder.finish()]);
  }
  
}

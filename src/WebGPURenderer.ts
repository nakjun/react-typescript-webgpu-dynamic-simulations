import { Particle } from "./particle";
import { mat4, vec3 } from 'gl-matrix';

export class WebGPURenderer {
    private canvas: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format: GPUTextureFormat = "bgra8unorm";
    private particles: Particle[];
    private pipeline!: GPURenderPipeline;
    private computePipeline!: GPUComputePipeline;
    private particleBuffer!: GPUBuffer;
    private queue!: GPUQueue;
    private computeBindGroup!: GPUBindGroup;

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
            const position: [number, number, number] = [
                Math.random() * 2 - 1, // Random position in range [-1, 1]
                Math.random() * 2 - 1,
                Math.random() * 2 - 1,
            ];
            const color: [number, number, number] = [
                Math.random(), // Random color
                Math.random(),
                Math.random(),
            ];
            const velocity: [number, number, number] = [0, 0, 0]; // Initial velocity
            this.particles.push(new Particle(position, color, velocity));
        }
    }

    createBuffers() {
        const particleData = new Float32Array(this.particles.length * 9);
        for (let i = 0; i < this.particles.length; i++) {
            particleData.set([...this.particles[i].position, ...this.particles[i].color, ...this.particles[i].velocity], i * 9);
        }

        this.particleBuffer = this.device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
        this.particleBuffer.unmap();
    }


    createComputePipeline() {
        const computeShader = `
    @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;

    struct Particle {
        position: vec3<f32>,
        color: vec3<f32>,
        velocity: vec3<f32>,
    };

    const gravity:vec3<f32> = vec3<f32>(0.0, -9.81, 0.0);
    const deltaTime: f32 = 0.001; // Assuming 60 FPS for simulation

    @compute @workgroup_size(64)
    fn cs_main(@builtin(global_invocation_id) id : vec3<u32>) {
        let i = id.x;
        if (i >= arrayLength(&particles)) {
            return;
        }

        

        // Update velocity based on gravity
        particles[i].velocity = particles[i].velocity + (gravity * deltaTime);

        // Update position based on velocity
        particles[i].position = particles[i].position + (particles[i].velocity * deltaTime);
    }
    `;


        const computeShaderModule = this.device.createShaderModule({
            code: computeShader,
        });

        // Create the bind group layout for the particle buffer
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                },
            }],
        });

        // Create the compute pipeline
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: computeShaderModule,
                entryPoint: 'cs_main',
            },
        });

        // Create the particle buffer
        // Assuming `this.particleBuffer` is already created and contains particles data
        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.particleBuffer,
                },
            }],
        });
    }


    dispatchComputeShader() {
        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        const numWorkgroups = Math.ceil(this.particles.length / 64);
        computePass.dispatchWorkgroups(numWorkgroups, 1, 1);
        computePass.end();

        this.queue.submit([commandEncoder.finish()]);
    }



    createPipeline() {
        // Shader Module
        const shaders = `struct VertexInput {
      @location(0) position : vec3<f32>,
      @location(1) color : vec3<f32>,
    };
  
    struct Fragment {
      @builtin(position) Position : vec4<f32>,
      @location(0) Color : vec4<f32>,
    };
  
    @vertex
    fn vs_main(vertexInput: VertexInput) -> Fragment {
      var output : Fragment;
      output.Position = vec4<f32>(vertexInput.position, 1.0);
      output.Color = vec4<f32>(vertexInput.color, 1.0);
      return output;
    }
  
    @fragment
    fn fs_main(@location(0) Color: vec4<f32>) -> @location(0) vec4<f32> {
      return Color;
    }`;

        console.log(shaders);

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
                        arrayStride: 12, // size of each position vertex (vec2<f32>)
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

    setCamera(){

        const cameraPosition = vec3.fromValues(0.0, 15.0, 10.0);
        const cameraTarget = vec3.fromValues(0, 0, 0);
        const cameraUp = vec3.fromValues(0, 1, 0);
        
        const viewMatrix = mat4.lookAt(mat4.create(), cameraPosition, cameraTarget, cameraUp);
        const projectionMatrix = mat4.perspective(mat4.create(), Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 100.0);

        // Create buffers for view and projection matrices
        const viewMatrixBuffer = this.device.createBuffer({
            size: 16 * 4, // 4x4 matrix, 16 bytes per float
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const projectionMatrixBuffer = this.device.createBuffer({
            size: 16 * 4, // 4x4 matrix, 16 bytes per float
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    render() {
        this.dispatchComputeShader();

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
        passEncoder.setVertexBuffer(0, this.particleBuffer);
        passEncoder.setVertexBuffer(1, this.particleBuffer, this.particles.length * 4 * 3);

        const emptyBindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [] });
        passEncoder.setBindGroup(0, emptyBindGroup);

        passEncoder.draw(this.particles.length, 1, 0, 0);
        passEncoder.end();

        this.queue.submit([commandEncoder.finish()]);
    }
}

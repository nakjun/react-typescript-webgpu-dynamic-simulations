// import { Particle } from "./particle";
import { mat4, vec3 } from 'gl-matrix';

export class WebGPURenderer {
    private canvas: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    
    private pipeline!: GPURenderPipeline;
    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private mvpUniformBuffer!: GPUBuffer;
    private renderBindGroup!: GPUBindGroup;
    
    private positions: number[];
    private velocities: number[];
    
    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;

    numParticles:number = 0;

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.positions = [];
        this.velocities = [];
        console.log("생성");
    }

    async init() {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) {
            throw new Error("Failed to get GPU adapter");
        }
        this.device = await adapter?.requestDevice();
        this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
        this.format = "bgra8unorm";
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque",
        });
    }

    async readBackPositionBuffer() {
        // Create a GPUBuffer for reading back the data
        const readBackBuffer = this.device.createBuffer({
            size: this.positionBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    
        // Create a command encoder and copy the position buffer to the readback buffer
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.positionBuffer, 0, readBackBuffer, 0, this.positionBuffer.size);
        
        // Submit the command to the GPU queue
        const commands = commandEncoder.finish();
        this.device.queue.submit([commands]);    
    
        // Map the readback buffer for reading and read its contents
        await readBackBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBackBuffer.getMappedRange(0, this.positionBuffer.size);
        const data = new Float32Array(arrayBuffer);
        console.log('Position Buffer Data:', data);
    
        // Cleanup
        readBackBuffer.unmap();
        readBackBuffer.destroy();
    }
    
    createParticles(numParticles: number) {
        this.numParticles = numParticles;
        for (let i = 0; i < numParticles; i++) {
            const position: [number, number, number] = [
                Math.random() * 10 - 5.0, // Random position in range [-1, 1]
                Math.random() * 10 + 5.0,
                0.0,
            ];
            const velocity: [number, number, number] = [0.0, 0.0, 0.0]; // Initial velocity
            this.positions.push(...position);
            this.velocities.push(...velocity);

            console.log(position, velocity);
        }
    }

    createBuffers() {
        // Position buffer creation
        const positionData = new Float32Array(this.positions);
        const bufferSize = this.numParticles * 3 * 4; // Correct size calculation

        this.positionBuffer = this.device.createBuffer({
            size: bufferSize, // Use the calculated size
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.positionBuffer.getMappedRange()).set(positionData);        
        this.positionBuffer.unmap();
    
        // Velocity buffer creation
        const velocityData = new Float32Array(this.velocities);
        this.velocityBuffer = this.device.createBuffer({
            size: velocityData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.velocityBuffer.getMappedRange()).set(velocityData); // Corrected to set velocityData
        this.velocityBuffer.unmap();

        this.readBackPositionBuffer();
    }

    createComputePipeline() {
        const computeShader = `
        @group(0) @binding(0) var<storage, read_write> positions : array<vec3<f32>>;
        @group(0) @binding(1) var<storage, read_write> velocities : array<vec3<f32>>;
    
        const gravity: vec3<f32> = vec3<f32>(0.0, -9.81, 0.0);
        const deltaTime: f32 = 0.01;

        @compute @workgroup_size(64)
        fn cs_main(@builtin(global_invocation_id) id : vec3<u32>) {
            let i = i32(id.x);

            if(positions[i].y <0.0){
                velocities[i] *=  -0.97;
                positions[i].y = 0.01;
            }
    
            // Simulate gravity effect on velocity            
            velocities[i] += (gravity * deltaTime);
    
            // Update position based on velocity
            positions[i] += (velocities[i] * deltaTime);
        }
        `;
    
        const computeShaderModule = this.device.createShaderModule({
            code: computeShader,
        });
    
        // Update the bind group layout to include both positions and velocities
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                    },
                },
                {
                    binding: 1, // Add this entry for the velocities buffer
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                    },
                }
            ],
        });
    
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: computeShaderModule,
                entryPoint: 'cs_main',
            },
        });
    
        // Assuming positionBuffer and velocityBuffer are already created and contain data
        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.positionBuffer,
                    },
                },
                {
                    binding: 1, // Ensure this matches the velocities buffer
                    resource: {
                        buffer: this.velocityBuffer,
                    },
                }
            ],
        });
    }
    

    dispatchComputeShader() {
        
    }

    createPipeline() {
        // Shader Module
        
        const shader = `struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

struct VertexInput {
    @location(0) position : vec3<f32>
};

struct FragmentOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) Color : vec4<f32>
};

@vertex
fn vs_main(vertexInput: VertexInput) -> FragmentOutput {
    var output : FragmentOutput;
    let modelViewProj = transformUBO.projection * transformUBO.view * transformUBO.model;
    output.Position = modelViewProj * vec4<f32>(vertexInput.position, 1.0);
    output.Color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Fixed color for all particles

    return output;
}

@fragment
fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
    return in.Color;
}

        `;

        const shaderModule = this.device.createShaderModule({ code: shader });

        // Bind Group Layout (if any resources need to be bound to the pipeline)
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {}
                }
            ]

        });

        // Attach mvp matrix uniform buffer
        this.mvpUniformBuffer = this.device.createBuffer({
            size: 64 * 3,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        
        this.renderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.mvpUniformBuffer
                    }
                }
            ]
        });

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
                buffers: [{
                    // Assuming position data is now solely in this buffer
                    arrayStride: 12, // 3 floats * 4 bytes per float
                    attributes: [{
                        // Position attribute
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x3"
                    }]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: "point-list",
            }
        });
    }

    setCamera(){
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4.0, this.canvas.width / this.canvas.height, 0.1, 1000.0);

        const view = mat4.create();
        mat4.lookAt(view, [0, 20, 150], [0, 0, 0], [0, 1, 0]);

        const model = mat4.create();          

        this.device.queue.writeBuffer(this.mvpUniformBuffer, 0, <ArrayBuffer>model); 
        this.device.queue.writeBuffer(this.mvpUniformBuffer, 64, <ArrayBuffer>view); 
        this.device.queue.writeBuffer(this.mvpUniformBuffer, 128, <ArrayBuffer>projection); 
    }

    async render() {
        const commandEncoder = this.device.createCommandEncoder();
    
        try {
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline); // Ensure this.computePipeline is a GPUComputePipeline
            computePass.setBindGroup(0, this.computeBindGroup);
            const numWorkgroups = Math.floor(this.numParticles / 64)+1;
            computePass.dispatchWorkgroups(numWorkgroups, 1, 1);
            computePass.end();
    
        } catch (error) {
            console.error("Error dispatching compute shader:", error);
        }
        
        // await this.device.queue.onSubmittedWorkDone();
        // await this.readBackPositionBuffer();

        const renderpass : GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: {r: 0.25, g: 0.25, b: 0.25, a: 1.0},
                loadOp: "clear",
                storeOp: "store"
            }]
        });

        this.setCamera();
        renderpass.setPipeline(this.pipeline);
        renderpass.setVertexBuffer(0, this.positionBuffer); // Use positionBuffer for rendering
        renderpass.setBindGroup(0, this.renderBindGroup);
        renderpass.draw(this.numParticles, 1, 0, 0); // Draw call matches the number of particles
        renderpass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
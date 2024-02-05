import { mat4, vec3 } from 'gl-matrix';
import { Camera } from './Camera';
import { Shader } from '../ParticleSystem/Shader';
import { Model } from '../Cube/Model';
import { RendererOrigin} from '../RendererOrigin';

export class Renderer extends RendererOrigin{    
    private particlePipeline!: GPURenderPipeline;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private mvpUniformBuffer!: GPUBuffer;
    private renderBindGroup!: GPUBindGroup;
    
    //shader
    private shader!: Shader;
    
    private positions!: number[];
    private velocities!: number[];
    
    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    numParticles:number = 0;

    constructor(canvasId: string) {
        super(canvasId);

        this.shader = new Shader();
        this.positions = [];
        this.velocities = [];
    }
    
    async init(){
        await super.init();
    }

    createParticlePipeline() {
        const particleShaderModule = this.device.createShaderModule({ code: this.shader.getParticleShader() });
        
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.VERTEX, // Accessible from the vertex shader
                    buffer: {} // Specifies that this binding will be a buffer
                }
            ]
        });
    
        // Create a uniform buffer for the MVP matrix. The size is 64 bytes * 3, assuming
        // you're storing three 4x4 matrices (model, view, projection) as 32-bit floats.
        // This buffer will be updated with the MVP matrix before each render.
        this.mvpUniformBuffer = this.device.createBuffer({
            size: 64 * 3, // The total size needed for the matrices
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST // The buffer is used as a uniform and can be copied to
        });
        
        // Create a bind group that binds the previously created uniform buffer to the shader.
        // This allows your shader to access the buffer as defined in the bind group layout.
        this.renderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout, // The layout created earlier
            entries: [
                {
                    binding: 0, // Corresponds to the binding in the layout
                    resource: {
                        buffer: this.mvpUniformBuffer // The buffer to bind
                    }
                }
            ]
        });
    
        // Create a pipeline layout that includes the bind group layouts.
        // This layout is necessary for the render pipeline to know how resources are structured.
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout], // Include the bind group layout created above
        });

        this.particlePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout, // Simplified layout, assuming no complex bindings needed
            vertex: {
                module: particleShaderModule,
                entryPoint: 'vs_main', // Ensure your shader has appropriate entry points
                buffers: [{
                    arrayStride: 12, // Assuming each particle position is a vec3<f32>
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
                }],                
            },
            fragment: {
                module: particleShaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'point-list', // Render particles as points
            },
            // Include depthStencil state if depth testing is required
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
        });
        console.log("create particle pipeline success");
    }

    createParticles(numParticles: number) {
        this.numParticles = numParticles;
        for (let i = 0; i < numParticles; i++) {
            const position: [number, number, number] = [
                Math.random() * 10 - 6, // Random position in range [-1, 1]
                Math.random() * 10 + 5.0,
                Math.random() * 10 - 5.0,
                // Math.random() * 10,
                // 5.0,
                // 0.0,
            ];
            const velocity: [number, number, number] = [0.0, 0.0, 0.0]; // Initial velocity
            this.positions.push(...position);
            this.velocities.push(...velocity);
        }
        console.log("create particle success #",numParticles);        
    }

    createParticleBuffers() {
        const positionData = new Float32Array(this.positions);
        this.positionBuffer = this.device.createBuffer({
            size: positionData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE, 
            mappedAtCreation: true,
        });
        new Float32Array(this.positionBuffer.getMappedRange()).set(positionData);
        this.positionBuffer.unmap();

        const velocityData = new Float32Array(this.velocities);
        this.velocityBuffer = this.device.createBuffer({
            size: velocityData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE, 
            mappedAtCreation: true,
        });
        new Float32Array(this.velocityBuffer.getMappedRange()).set(velocityData);
        this.velocityBuffer.unmap();
    }

    createComputeBindGroup() {
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.positionBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.velocityBuffer,
                    },
                },
            ],
        });
    }

    createComputePipeline() {        
        const computeShaderModule = this.device.createShaderModule({ code: this.shader.getComputeShader() });
    
        // Create bind group layout for storage buffers
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // matches @group(0) @binding(0)
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 1, // matches @group(0) @binding(1)
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
            ],
        });
    
        // Use the bind group layout to create a pipeline layout
        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
        const computePipeline = this.device.createComputePipeline({
            layout: computePipelineLayout, // Use the created pipeline layout
            compute: {
                module: computeShaderModule,
                entryPoint: 'main',
            },
        });
    
        this.computePipeline = computePipeline;
    }

    setCamera(camera: Camera) {
        // Projection matrix: Perspective projection
        const projection = mat4.create();
        mat4.perspective(projection, camera.fov, this.canvas.width / this.canvas.height, camera.near, camera.far);
    
        // View matrix: Camera's position and orientation in the world
        const view = mat4.create();
        mat4.lookAt(view, camera.position, camera.target, camera.up);
    
        // Model matrix: For now, we can use an identity matrix if we're not transforming the particles
        const model = mat4.create(); // No transformation to the model
    
        // Now, update the buffer with these matrices
        this.updateUniformBuffer(model, view, projection);
    }

    updateParticles() {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64.0)+1, 1, 1);
        computePass.end();
    }
    

    updateUniformBuffer(model: mat4, view: mat4, projection: mat4) {
        // Combine the matrices into a single Float32Array
        const data = new Float32Array(48); // 16 floats per matrix, 3 matrices
        data.set(model);
        data.set(view, 16); // Offset by 16 floats for the view matrix
        data.set(projection, 32); // Offset by 32 floats for the projection matrix
    
        // Upload the new data to the GPU
        this.device.queue.writeBuffer(
            this.mvpUniformBuffer,
            0, // Start at the beginning of the buffer
            data.buffer, // The ArrayBuffer of the Float32Array
            0, // Start at the beginning of the data
            data.byteLength // The amount of data to write
        );
    }
    async render() {        
        
        const currentTime = performance.now();
        this.frameCount++;
        
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64.0)+1, 1, 1);
        computePass.end();
        
        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, // Background color
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: { // Add this attachment for depth testing
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        };
    
        // Set camera and model transformations here
        // Assuming setCamera() updates the uniform buffer with the MVP matrix
        this.setCamera(this.camera);    

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer        
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.draw(this.numParticles); // Draw the cube using the index count
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);

        if (currentTime - this.lastTime >= 1000) {
            // Calculate the FPS.
            const fps = this.frameCount;
    
            // Optionally, display the FPS in the browser.
            if (this.fpsDisplay) {
                this.fpsDisplay.textContent = `FPS: ${fps}`;
            } else {
                console.log(`FPS: ${fps}`);
            }
    
            // Reset the frame count and update the last time check.
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }
    
}
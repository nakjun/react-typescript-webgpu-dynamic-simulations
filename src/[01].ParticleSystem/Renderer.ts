import { mat4, vec3 } from 'gl-matrix';
import { Camera } from '../WebGPU/Camera';
import { Shader } from './Shader';
import { RendererOrigin} from '../RendererOrigin';

export class Renderer extends RendererOrigin{    
    private particlePipeline!: GPURenderPipeline;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    
    private renderBindGroup!: GPUBindGroup;
    private numParticlesBuffer!: GPUBuffer;
    
    //shader
    private shader!: Shader;
    
    private positions!: number[];
    private velocities!: number[];    
    private colors!: number[];
    
    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private colorBuffer!: GPUBuffer;
    numParticles:number = 0;

    bottomPanel!:Float32Array;
    bottomPanelIndicies!:Uint16Array;
    indexCount:number = 0;

    private vertexBuffer!:GPUBuffer;
    private indexBuffer!:GPUBuffer;
    

    constructor(canvasId: string) {
        super(canvasId);

        this.shader = new Shader();
        this.positions = [];
        this.velocities = [];
        this.colors = [];

        this.bottomPanel = new Float32Array([
            -20.0, -0.01, -20.0, 0.0, 0.0, 1.0,
            20.0, -0.01, -20.0, 0.0, 0.0, 1.0,
            20.0, -0.01, 20.0, 0.0, 0.0, 1.0,
            -20.0, -0.01, 20.0, 0.0, 0.0, 1.0,
        ]);

        this.bottomPanelIndicies = new Uint16Array([
            0, 1, 2, 0, 2, 3,
        ]);

    }
    
    async init(){
        await super.init();
    }

    createBuffers() {                
        // Create vertex buffer
        this.vertexBuffer = this.device.createBuffer({
            size: this.bottomPanel.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.bottomPanel);
        this.vertexBuffer.unmap();
    
        // Create index buffer
        this.indexBuffer = this.device.createBuffer({
            size: this.bottomPanelIndicies.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint16Array(this.indexBuffer.getMappedRange()).set(this.bottomPanelIndicies);
        this.indexBuffer.unmap();
    
        // Set the count of indices
        this.indexCount = this.bottomPanelIndicies.length;
    }
    
    createPipeline(){
        // Load the shader code for rendering, presumably from a Shader class instance
        // This shader code likely contains both the vertex and fragment shaders.
        const render_shader = this.shader.getRenderShader();
    
        // Create a shader module from the shader code. This module is used by the GPU
        // to execute the shader code during rendering.
        const shaderModule = this.device.createShaderModule({ code: render_shader });
    
        // Define a bind group layout. This layout specifies the resources (like uniform buffers,
        // storage buffers, samplers, etc.) that will be accessible by the shader.
        // Here, it's set up for a uniform buffer (likely for model-view-projection matrix) that
        // will be used in the vertex shader.
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
    
        // Finally, create the render pipeline. This pipeline configures the GPU on how to render
        // your data using the shaders, bind groups, and buffers you've defined.
        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout, // The pipeline layout
            vertex: {
                module: shaderModule, // The shader module for the vertex shader
                entryPoint: "vs_main", // The entry point function in the shader code
                buffers: [{ // Define the layout of the vertex data
                    arrayStride: 24, // Each vertex consists of 3 floats (x, y, z), each float is 4 bytes
                    attributes: [{ // Describe the attributes of the vertex data
                        shaderLocation: 0, // Corresponds to the location in the vertex shader
                        offset: 0, // The offset within the buffer (start at the beginning)
                        format: "float32x3" // The format of the vertex data (3-component vector of 32-bit floats)
                    },
                    { // Describe the attributes of the vertex data
                        shaderLocation: 1, // Corresponds to the location in the vertex shader
                        offset: 12, // The offset within the buffer (start at the beginning)
                        format: "float32x3" // The format of the vertex data (3-component vector of 32-bit floats)
                    }]
                    
                }]
            },
            fragment: {
                module: shaderModule, // The shader module for the fragment shader
                entryPoint: "fs_main", // The entry point function in the shader code
                targets: [{ format: this.format }], // The format of the rendering target, must match the swap chain's format
            },
            primitive: {
                topology: "triangle-list", // The type of primitive to render. "point-list" means each vertex is a separate point.
            },
            depthStencil: { // This part needs to be added or corrected
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float', // Make sure this matches the depth texture's format
            }
    
        });
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
                },
                {
                    arrayStride: 12, // Assuming each particle position is a vec3<f32>
                    attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
                }
            ],                
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

    random(min:number, max:number) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
      }

    createParticles(numParticles: number) {
        this.numParticles = numParticles;
        for (let i = 0; i < numParticles; i++) {
            const position: [number, number, number] = [
                this.random(-20.0,20.0),
                this.random(5.0,30.0),
                this.random(-20.0,20.0),
            ];
            const color: [number, number, number] = [
                Math.random() * 10.0,
                Math.random() * 10.0,
                Math.random() * 10.0,
            ]
            
            const velocity: [number, number, number] = [0.0, 0.0, 0.0]; // Initial velocity
            this.positions.push(...position);
            this.velocities.push(...velocity);
            this.colors.push(...color);
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
            usage: GPUBufferUsage.STORAGE, 
            mappedAtCreation: true,
        });
        new Float32Array(this.velocityBuffer.getMappedRange()).set(velocityData);
        this.velocityBuffer.unmap();

        const colorData = new Float32Array(this.colors);
        this.colorBuffer = this.device.createBuffer({
            size: colorData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 
            mappedAtCreation: true,
        });
        new Float32Array(this.colorBuffer.getMappedRange()).set(colorData);
        this.colorBuffer.unmap();

        const numParticlesData = new Uint32Array([this.numParticles]);
        this.numParticlesBuffer = this.device.createBuffer({
            size: numParticlesData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.numParticlesBuffer.getMappedRange()).set(numParticlesData);
        this.numParticlesBuffer.unmap();
    }

    createComputeBindGroup() {
        
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
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                },
            ],
        });
    
        // Use the bind group layout to create a pipeline layout
        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
        this.computePipeline = this.device.createComputePipeline({
            layout: computePipelineLayout, // Use the created pipeline layout
            compute: {
                module: computeShaderModule,
                entryPoint: 'main',
            },
        });

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
                    binding: 1,
                    resource: {
                        buffer: this.velocityBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.numParticlesBuffer,
                    },
                },
            ],
        });
    }

    

    updateParticles() {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);

        let X = Math.ceil(this.numParticles / 64);

        computePass.dispatchWorkgroups(X, 1, 1);
        computePass.end();
    }
    

    
    async render() {        
        
        const currentTime = performance.now();
        this.frameCount++;
        this.setCamera(this.camera);    
        
        {
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64.0)+1);
            computePass.end();        
            this.device.queue.submit([commandEncoder.finish()]);
        }
        
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
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer        
        passEncoder.setVertexBuffer(1, this.colorBuffer); // Set the vertex buffer        
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.draw(this.numParticles); // Draw the cube using the index count

        passEncoder.setPipeline(this.pipeline); // Your render pipeline
        passEncoder.setVertexBuffer(0, this.vertexBuffer); // Set the vertex buffer
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16'); // Set the index buffer
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.drawIndexed(this.indexCount); // Draw the cube using the index count
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
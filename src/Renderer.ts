import { mat4, vec3 } from 'gl-matrix';
import { Camera } from './Camera';
import { Shader } from './Shader';
import { Model } from './Model';

export class Renderer{
    private canvas!: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    private depthTexture!: GPUTexture;
    
    private pipeline!: GPURenderPipeline;
    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private mvpUniformBuffer!: GPUBuffer;
    private renderBindGroup!: GPUBindGroup;
    
    
    //axis render
    private axisColorVertices!: Float32Array;
    private axisVertexBuffer!: GPUBuffer;
    
    //shader
    private shader!: Shader;
    
    //camera
    private camera!: Camera;
    
    //model
    private cubeModel!: Model;
    private positions!: number[];
    private velocities!: number[];
    
    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;

    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private indexCount:number = 0;

    numParticles:number = 0;
    
    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.positions = [];
        this.velocities = [];
        this.shader = new Shader();
        console.log("Renderer initialized");
        this.cubeModel = new Model();
        this.camera = new Camera(
            vec3.fromValues(0, 3, 5), // position
            vec3.fromValues(0, 0, 0), // target
            vec3.fromValues(0, 1, 0), // up
            Math.PI / 4, // fov in radians
            this.canvas.width / this.canvas.height, // aspect ratio
            0.1, // near
            100 // far
        );
    }

    async init() {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) {
            throw new Error("Failed to get GPU adapter");
        }
        this.device = await adapter?.requestDevice();
        this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;
        this.format = "bgra8unorm";
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque",
        });
        this.createDepthTexture();
    }
    
    createDepthTexture() {
        const depthTexture = this.device.createTexture({
            size: { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 },
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTexture = depthTexture;
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
        // Create vertex buffer
        this.vertexBuffer = this.device.createBuffer({
            size: this.cubeModel.get_cubeVertices().byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.cubeModel.get_cubeVertices());
        this.vertexBuffer.unmap();

        // Create index buffer
        this.indexBuffer = this.device.createBuffer({
            size: this.cubeModel.get_cubeIndices().byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint16Array(this.indexBuffer.getMappedRange()).set(this.cubeModel.get_cubeIndices());
        this.indexBuffer.unmap();

        // Set the count of indices
        this.indexCount = this.cubeModel.get_cubeIndices().length;
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
        const commandEncoder = this.device.createCommandEncoder();
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
        passEncoder.setPipeline(this.pipeline); // Your render pipeline
        passEncoder.setVertexBuffer(0, this.vertexBuffer); // Set the vertex buffer
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16'); // Set the index buffer
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.drawIndexed(this.indexCount); // Draw the cube using the index count
        passEncoder.end();
    
        this.device.queue.submit([commandEncoder.finish()]);
    }
    
}
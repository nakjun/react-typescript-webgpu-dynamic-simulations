import { mat4, vec3 } from 'gl-matrix';
import { Camera } from '../WebGPU/Camera';
import { Shader } from '../ParticleSystem/Shader';
import { Model } from '../Cube/Model';

export class CubeRender{

    private canvas!: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    private depthTexture!: GPUTexture;
    private pipeline!: GPURenderPipeline;

    //shader
    private shader!: Shader;

    //model
    private cubeModel!: Model;

    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private indexCount:number = 0;

    private mvpUniformBuffer!: GPUBuffer;
    private renderBindGroup!: GPUBindGroup;

    //camera
    private camera!: Camera;
    camera_position:vec3 = vec3.fromValues(0.0, 0.0, 15.0);
    camera_target:vec3 = vec3.fromValues(0.0, 0.0, 0.0);
    camera_up:vec3 = vec3.fromValues(0.0, 1.0, 0.0);

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;        
        this.shader = new Shader();
        console.log("Renderer initialized");
        this.cubeModel = new Model();
        this.camera = new Camera(
            this.camera_position, // position
            this.camera_target, // target
            this.camera_up, // up
            Math.PI / 4, // fov in radians
            this.canvas.width / this.canvas.height, // aspect ratio
            0.1, // near
            100 // far
        );
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

}


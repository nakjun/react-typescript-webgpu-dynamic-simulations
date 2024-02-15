import { mat4, vec3 } from 'gl-matrix';
import { Camera } from './WebGPU/Camera';
import { Shader } from './[01].ParticleSystem/Shader';
import { Model } from './Common/Model';
import { SystemGUI } from './GUI/GUI';

export class RendererOrigin {

    canvas!: HTMLCanvasElement;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;
    depthTexture!: GPUTexture;
    pipeline!: GPURenderPipeline;
    mvpUniformBuffer!: GPUBuffer;

    systemGUI!:SystemGUI;

    //camera
    camera!: Camera;
    camera_position: vec3 = vec3.fromValues(-91.0, 39.0, 37.0);
    camera_target: vec3 = vec3.fromValues(0.0, 0.0, 0.0);
    camera_up: vec3 = vec3.fromValues(0.0, 1.0, 0.0);

    //fps
    frameCount: number = 0;
    lastTime: number = 0;
    fpsDisplay;
    localFrameCount:number =0;

    stats = {
        fps: 0,
        ms:""
    };

    renderOptions = {
        wireFrame: false,
    }

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.camera = new Camera(
            this.camera_position, // position
            this.camera_target, // target
            this.camera_up, // up
            Math.PI / 4, // fov in radians
            this.canvas.width / this.canvas.height, // aspect ratio
            0.1, // near
            10000 // far
        );
        console.log("Renderer initialized");
        this.fpsDisplay = document.getElementById('fpsDisplay');

        this.systemGUI = new SystemGUI();        
        this.systemGUI.performanceGui.add(this.stats, 'ms').name('ms').listen();
        this.systemGUI.performanceGui.add(this.stats, 'fps').name('fps').listen();

        this.systemGUI.renderOptionGui.add(this.renderOptions, 'wireFrame').name('WireFrame');

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
        this.depthTexture = this.device.createTexture({
            size: { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 },
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
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

    rotateCamera(dx:number, dy:number){
        this.camera.position[0] += dx;
        this.camera.position[1] += dy;
        console.log(this.camera_position);
    }

    panCamera(dx:number, dy:number){

    }

    zoomCamera(value:number){
        this.camera_position[2] += value;
        console.log(this.camera_position);
    }
}
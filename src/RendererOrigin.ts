import { mat4, vec3 } from 'gl-matrix';
import { Camera } from './WebGPU/Camera';
import { Shader } from './ParticleSystem/Shader';
import { Model } from './Cube/Model';

export class RendererOrigin {

    canvas!: HTMLCanvasElement;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;
    depthTexture!: GPUTexture;
    pipeline!: GPURenderPipeline;

    //camera
    camera!: Camera;
    camera_position: vec3 = vec3.fromValues(1.0, 5.0, 10.0);
    camera_target: vec3 = vec3.fromValues(0.0, 0.0, 0.0);
    camera_up: vec3 = vec3.fromValues(0.0, 1.0, 0.0);

    //fps
    frameCount: number = 0;
    lastTime: number = 0;
    fpsDisplay;

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.camera = new Camera(
            this.camera_position, // position
            this.camera_target, // target
            this.camera_up, // up
            Math.PI / 4, // fov in radians
            this.canvas.width / this.canvas.height, // aspect ratio
            0.1, // near
            100 // far
        );
        console.log("Renderer initialized");
        this.fpsDisplay = document.getElementById('fpsDisplay');
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
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
}
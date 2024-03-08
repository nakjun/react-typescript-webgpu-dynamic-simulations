import { mat4, vec3 } from 'gl-matrix';
import { Camera } from './WebGPU/Camera';
import { SystemGUI } from './GUI/GUI';

export class RendererOrigin {

    canvas!: HTMLCanvasElement;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;
    depthTexture!: GPUTexture;
    resolveTexture!: GPUTexture;
    pipeline!: GPURenderPipeline;
    mvpUniformBuffer!: GPUBuffer;
    sampleCount: number = 4;

    //camera
    camera!: Camera;
    camera_position: vec3 = vec3.fromValues(100, 25, -30);
    camera_target: vec3 = vec3.fromValues(-0.13, 8.48, -1.76);
    camera_up: vec3 = vec3.fromValues(0.0, 1.0, 0.0);

    //lighting
    light_position: vec3 = vec3.fromValues(20.0, 150.0, -20);
    light_color: vec3 = vec3.fromValues(1.0, 1.0, 1.0);
    light_intensity: number = 0.65;
    specular_strength: number = 1.5;
    shininess: number = 2048.0;

    //fps
    frameCount: number = 0;
    lastTime: number = 0;
    fpsDisplay;
    localFrameCount: number = 0;

    //gui
    systemGUI!: SystemGUI;
    stats = {
        fps: 0,
        ms: ""
    };

    camPosXControl: any;
    camPosYControl: any;
    camPosZControl: any;

    lightPosXControl: any;
    lightPosYControl: any;
    lightPosZControl: any;

    lightColorXControl: any;
    lightColorYControl: any;
    lightColorZControl: any;

    renderOptions = {
        wireFrame: false,
        camPosX: this.camera_position[0],
        camPosY: this.camera_position[1],
        camPosZ: this.camera_position[2],
        renderObject: true,
        moveObject: false,
        wind: false,

        lightPosX: this.light_position[0],
        lightPosY: this.light_position[1],
        lightPosZ: this.light_position[2],

        lightColorX: this.light_color[0],
        lightColorY: this.light_color[1],
        lightColorZ: this.light_color[2],

        lightIntensity: this.light_intensity,
        specularStrength: this.specular_strength,
        shininess: this.shininess,
    }

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.camera = new Camera(
            this.camera_position,
            this.camera_target,
            this.camera_up,
            Math.PI / 4,
            this.canvas.width / this.canvas.height,
            0.1,
            10000
        );
        console.log("Renderer initialized");
        this.fpsDisplay = document.getElementById('fpsDisplay');

        this.systemGUI = new SystemGUI();
        this.systemGUI.performanceGui.add(this.stats, 'ms').name('ms').listen();
        this.systemGUI.performanceGui.add(this.stats, 'fps').name('fps').listen();

        this.systemGUI.renderOptionGui.add(this.renderOptions, 'wireFrame').name('WireFrame');
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'renderObject').name('renderObject');
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'moveObject').name('moveObject');
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'wind').name('Apply Wind');
        this.camPosXControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'camPosX', -500, 500).name('Camera Position X').onChange((value: number) => {
            this.camera.position[0] = value;
        });
        this.camPosYControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'camPosY', -500, 500).name('Camera Position Y').onChange((value: number) => {
            this.camera.position[1] = value;
        });
        this.camPosZControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'camPosZ', -500, 500).name('Camera Position Z').onChange((value: number) => {
            this.camera.position[2] = value;
        });
        this.lightPosXControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightPosX', -500, 500).name('Light Position X').onChange((value: number) => {
            this.light_position[0] = value;
        });
        this.lightPosYControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightPosY', -500, 500).name('Light Position Y').onChange((value: number) => {
            this.light_position[1] = value;
        });
        this.lightPosZControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightPosZ', -500, 500).name('Light Position Z').onChange((value: number) => {
            this.light_position[2] = value;
        });
        this.lightColorXControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightColorX', 0.0, 1.0).step(0.01).name('Light Color X').onChange((value: number) => {
            this.light_color[0] = value;
        });
        this.lightColorYControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightColorY', 0.0, 1.0).step(0.01).name('Light Color Y').onChange((value: number) => {
            this.light_color[1] = value;
        });
        this.lightColorZControl = this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightColorZ', 0.0, 1.0).step(0.01).name('Light Color Z').onChange((value: number) => {
            this.light_color[2] = value;
        });
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'lightIntensity', 0.0, 10.0).step(0.01).name('Light Intensity').onChange((value: number) => {
            this.light_intensity = value;
        });
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'specularStrength', 0.0, 10.0).step(0.01).name('Specular Strength').onChange((value: number) => {
            this.specular_strength = value;
        });
        this.systemGUI.renderOptionGui.add(this.renderOptions, 'shininess', 0.0, 100000.0).step(1).name('Shininess').onChange((value: number) => {
            this.shininess = value;
        });
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
        this.createResolveTexture();
        this.printDeviceLimits();

    }
    printDeviceLimits() {
        const limits: GPUSupportedLimits = this.device.limits;
        this.systemGUI.gpuDeviceGui.add(limits, 'maxComputeWorkgroupSizeX').name('Max Compute Workgroup Size X');
        this.systemGUI.gpuDeviceGui.add(limits, 'maxComputeWorkgroupSizeY').name('Max Compute Workgroup Size Y');
        this.systemGUI.gpuDeviceGui.add(limits, 'maxComputeWorkgroupSizeZ').name('Max Compute Workgroup Size Z');
        this.systemGUI.gpuDeviceGui.add(limits, 'maxComputeInvocationsPerWorkgroup').name('Max Compute Invocations Per Workgroup');
        this.systemGUI.gpuDeviceGui.add(limits, 'maxComputeWorkgroupsPerDimension').name('Max Compute Workgroups Per Dimension');
        this.systemGUI.gpuDeviceGui.add(limits, 'maxStorageBufferBindingSize').name('Max Storage Buffer Binding Size');
    }

    createDepthTexture() {
        this.depthTexture = this.device.createTexture({
            size: { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 },
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            sampleCount: this.sampleCount
        });
    }

    createResolveTexture() {
        this.resolveTexture = this.device.createTexture({
            size: { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 },
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            sampleCount: this.sampleCount
        });
    }

    setCamera(camera: Camera) {
        const projection = mat4.create();
        mat4.perspective(projection, camera.fov, this.canvas.width / this.canvas.height, camera.near, camera.far);

        const view = mat4.create();
        mat4.lookAt(view, camera.position, camera.target, camera.up);

        const model = mat4.create();

        this.updateUniformBuffer(model, view, projection);
    }

    updateUniformBuffer(model: mat4, view: mat4, projection: mat4) {
        const data = new Float32Array(48);
        data.set(model);
        data.set(view, 16);
        data.set(projection, 32);

        this.device.queue.writeBuffer(
            this.mvpUniformBuffer,
            0,
            data.buffer,
            0,
            data.byteLength
        );
    }

    updateRenderOptions() {
        this.renderOptions.camPosX = this.camera.position[0];
        this.renderOptions.camPosY = this.camera.position[1];
        this.renderOptions.camPosZ = this.camera.position[2];

        this.camPosXControl.updateDisplay();
        this.camPosYControl.updateDisplay();
        this.camPosZControl.updateDisplay();

        this.printCameraValue();
    }

    rotateCamera(dx: number, dy: number) {
        this.camera.position[0] += dx;
        this.camera.position[1] += dy;

        this.updateRenderOptions();
    }

    panCamera(dx: number, dy: number) {
        const cameraDirection = vec3.subtract(vec3.create(), this.camera.target, this.camera.position);
        const cameraRight = vec3.cross(vec3.create(), cameraDirection, this.camera.up);
        const cameraUp = vec3.cross(vec3.create(), cameraRight, cameraDirection);

        vec3.normalize(cameraRight, cameraRight);
        vec3.normalize(cameraUp, cameraUp);

        const scale = 0.1;

        vec3.scaleAndAdd(this.camera.position, this.camera.position, cameraRight, dx * scale);
        vec3.scaleAndAdd(this.camera.position, this.camera.position, cameraUp, -dy * scale);

        vec3.scaleAndAdd(this.camera.target, this.camera.target, cameraRight, dx * scale);
        vec3.scaleAndAdd(this.camera.target, this.camera.target, cameraUp, -dy * scale);

        this.updateRenderOptions();
    }

    zoomCamera(value: number) {
        this.camera.position[2] += value;

        this.updateRenderOptions();
    }

    printCameraValue() {
        console.log(this.camera.position);
        console.log(this.camera.target);
        console.log(this.camera.up);
    }
}
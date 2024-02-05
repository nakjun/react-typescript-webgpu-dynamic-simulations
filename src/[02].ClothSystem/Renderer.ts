import { mat4, vec3 } from 'gl-matrix';
import { Camera } from '../WebGPU/Camera';
import { ParticleShader } from './ParticleShader';
import { SpringShader } from './SpringShader';
import { RendererOrigin } from '../RendererOrigin';

class Node {
    position!: vec3;
    velocity!: vec3;
    acceleration!: vec3;

    fixed: boolean = false;

    constructor(pos: vec3, vel: vec3) {
        this.position = pos;
        this.velocity = vel;
        this.acceleration = vec3.create();
        this.fixed = false;
    }
}

class Spring {
    n1!: Node;
    n2!: Node;
    mRestLen: number = 0;

    kS: number = 1000.0;
    kD: number = 0.01;
    type: string = "spring type";

    constructor(_n1: Node, _n2: Node, ks: number, kd: number, type: string) {
        this.n1 = _n1;
        this.n2 = _n2;

        this.kS = ks;
        this.kD = kd;
        this.type = type;

        this.mRestLen = vec3.distance(this.n1.position, this.n2.position);
    }
}

export class ClothRenderer extends RendererOrigin {

    private particlePipeline!: GPURenderPipeline;
    private springPipeline!: GPURenderPipeline;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    private numParticlesBuffer!: GPUBuffer;

    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private springBuffer!: GPUBuffer;

    //shader
    private particleShader!: ParticleShader;
    private springShader!: SpringShader;

    //particle information
    private particles: Node[] = [];
    private springs: Spring[] = [];

    numParticles: number = 0;

    //cloth information
    N: number = 0;
    M: number = 0;
    kS: number = 0;
    kD: number = 0;

    xSize:number = 30.0;
    ySize:number = 30.0;

    constructor(canvasId: string) {
        super(canvasId);
        this.particleShader = new ParticleShader();
        this.springShader = new SpringShader();
    }

    async init() {
        await super.init();
        
    }

    createClothModel(x:number, y:number, ks:number, kd:number){
        
        this.N = x;
        this.M = y;
        this.kS = ks;
        this.kD = kd;
        
        this.createParticles();
        this.createSprings();
    }

    createParticles() {
        // N * M 그리드의 노드를 생성하는 로직
        const start_x = -(this.xSize / 2.0);
        const start_y = this.ySize;

        const dist_x = (this.xSize / this.N);
        const dist_y = (this.ySize / this.M);

        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.M; j++) {
                var pos = vec3.fromValues(start_x + (dist_x * j), start_y - (dist_y * i), 0.0);
                var vel = vec3.fromValues(0, 0, 0);

                const n = new Node(pos, vel);

                this.particles.push(n);

                console.log(pos);
            }
        }
        console.log("create node success");
    }
    createSprings() {
        let index = 0;
        for (let i = 0; i < this.N - 1; i++) {
            for (let j = 0; j < this.M; j++) {                
                const spring = new Spring(
                    this.particles[index],
                    this.particles[index + 1],
                    this.kS,
                    this.kD,
                    "structural"
                );
                this.springs.push(spring);
            }
        }
        // 2. Structural 가로
        for (let i = 0; i < (this.M - 1); i++) {
            for (let j = 0; j < this.N; j++) {
                ++index;
                let sp = new Spring(this.particles[this.N * i + j], this.particles[this.N * i + j + this.N], this.kS, this.kD, "structural");
                this.springs.push(sp);
            }
        }
        // 3. Shear 우상좌하
        index = 0;
        for (let i = 0; i < (this.N) * (this.M - 1); i++) {
            if (i % this.N === (this.N - 1)) {
                index++;
                continue;
            }
            let sp = new Spring(this.particles[index], this.particles[index + this.N + 1], this.kS, this.kD, "shear");
            this.springs.push(sp);
            index++;
        }
        // 4. Shear 좌상우하
        index = 0;
        for (let i = 0; i < (this.N) * (this.M - 1); i++) {
            if (i % this.N === 0) {
                index++;
                continue;
            }
            let sp = new Spring(this.particles[index], this.particles[index + this.N - 1], this.kS, this.kD, "shear");
            this.springs.push(sp);
            index++;
        }
        // 5. Bending 가로
        index = 0;
        for (let i = 0; i < (this.N) * this.M; i++) {
            if (i % this.N > this.N - 3) {
                index++;
                continue;
            }
            let sp = new Spring(this.particles[index], this.particles[index + 2], this.kS, this.kD, "bending");
            this.springs.push(sp);
            index++;
        }
        //6. Bending 세로
        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.M - 3; j++) {
                let sp = new Spring(this.particles[i + (j * this.M)], this.particles[i + (j + 3) * this.M], this.kS, this.kD, "bending");
                this.springs.push(sp);                
            }
        }
    }

    createClothBuffers()
    {
        const positionData = new Float32Array(this.particles.flatMap(p => [p.position[0], p.position[1], p.position[2]]));
        const velocityData = new Float32Array(this.particles.flatMap(p => [p.velocity[0], p.velocity[1], p.velocity[2]]));

        this.positionBuffer = this.device.createBuffer({
            size: positionData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.positionBuffer.getMappedRange()).set(positionData);
        this.positionBuffer.unmap();
    
        this.velocityBuffer = this.device.createBuffer({
            size: velocityData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.velocityBuffer.getMappedRange()).set(velocityData);
        this.velocityBuffer.unmap();    

        const springData = new Float32Array(this.springs.flatMap(s => [
            ...s.n1.position, // Start position of the spring
            ...s.n2.position  // End position of the spring
        ]));
    
        this.springBuffer = this.device.createBuffer({
            size: springData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.springBuffer.getMappedRange()).set(springData);
        this.springBuffer.unmap();
    }

    createParticlePipeline() {
        const particleShaderModule = this.device.createShaderModule({ code: this.particleShader.getParticleShader() });
        
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

    createSpringPipeline() {
        const springShaderModule = this.device.createShaderModule({ code: this.particleShader.getSpringShader() });
    
        // Assuming bindGroupLayout and pipelineLayout are similar to createParticlePipeline
        // You may reuse the same layout if it fits your needs
    
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.VERTEX, // Accessible from the vertex shader
                    buffer: {} // Specifies that this binding will be a buffer
                }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout], // Include the bind group layout created above
        });

        this.springPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout, // Reuse or create as needed
            vertex: {
                module: springShaderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 12, // vec3<f32> for spring start and end positions
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
                }],
            },
            fragment: {
                module: springShaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'line-list',
                // Additional configurations as needed
            },
            // Reuse depthStencil configuration
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
        });
    }

    async render() {        
        const currentTime = performance.now();
        this.frameCount++;

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
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer                
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.draw(this.N * this.M); // Draw the cube using the index count
        
        passEncoder.setPipeline(this.springPipeline);
        passEncoder.setVertexBuffer(0, this.springBuffer);
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.draw(this.springs.length * 2);
        
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
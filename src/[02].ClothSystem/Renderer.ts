import { vec3 } from 'gl-matrix';
import { ParticleShader } from './ParticleShader';
import { SpringShader } from './SpringShader';
import { RendererOrigin } from '../RendererOrigin';

class Node {
    position!: vec3;
    velocity!: vec3;
    acceleration!: vec3;

    fixed: boolean = false;

    springs: Spring[] = [];

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

    index1: number = 0;
    index2: number = 0;

    constructor(_n1: Node, _n2: Node, ks: number, kd: number, type: string, _i1: number, _i2: number) {
        this.n1 = _n1;
        this.n2 = _n2;

        this.kS = ks;
        this.kD = kd;
        this.type = type;

        this.mRestLen = vec3.distance(this.n1.position, this.n2.position);

        this.index1 = _i1;
        this.index2 = _i2;
    }
}

export class ClothRenderer extends RendererOrigin {

    private particlePipeline!: GPURenderPipeline;
    private springPipeline!: GPURenderPipeline;
    private renderBindGroup!: GPUBindGroup;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private computeSpringPipeline!: GPUComputePipeline;
    private computeSpringBindGroup!: GPUBindGroup;
    private computeSpringRenderPipeline!: GPUComputePipeline;
    private computeSpringRenderBindGroup!: GPUBindGroup;

    private numParticlesBuffer!: GPUBuffer;
    private numSpringsBuffer!: GPUBuffer;

    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private fixedBuffer!: GPUBuffer;
    private springRenderBuffer!: GPUBuffer;
    private springCalculationBuffer!: GPUBuffer;

    //shader
    private particleShader!: ParticleShader;
    private springShader!: SpringShader;

    //particle information
    private particles: Node[] = [];
    private springs: Spring[] = [];
    private springIndicies!:Uint32Array;

    numParticles: number = 0;

    renderPassDescriptor!: GPURenderPassDescriptor;

    //cloth information
    N: number = 0;
    M: number = 0;
    kS: number = 0;
    kD: number = 0;

    xSize: number = 30.0;
    ySize: number = 30.0;

    //for temp storage buffer
    maxSpringConnected:number = 0;

    constructor(canvasId: string) {
        super(canvasId);
        this.particleShader = new ParticleShader();
        this.springShader = new SpringShader();
    }

    async init() {
        await super.init();        
    }

    createClothModel(x: number, y: number, ks: number, kd: number) {

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
            }
        }

        for(let i=0;i<this.N;i++){
            this.particles[i].fixed = true;
        }

        this.numParticles = this.particles.length;
        console.log("create node success");
    }
    createSprings() {
        let index = 0;
        for (let i = 0; i < this.M; i++) {
            for (let j = 0; j < this.N-1; j++) {
                if(i>0 && j===0) index++;
                const sp = new Spring(
                    this.particles[index],
                    this.particles[index + 1],
                    this.kS,
                    this.kD,
                    "structural",
                    index,
                    index + 1
                );
                this.springs.push(sp);
                this.particles[index].springs.push(sp);
                this.particles[index + 1].springs.push(sp);
                index++;
            }
        }
        // 2. Structural 세로
        for (let i = 0; i < (this.N - 1); i++) {
            for (let j = 0; j < this.M; j++) {
                ++index;
                const sp = new Spring(
                    this.particles[this.N * i + j], 
                    this.particles[this.N * i + j + this.N], 
                    this.kS, 
                    this.kD, 
                    "structural", 
                    this.N * i + j, 
                    this.N * i + j + this.N
                );
                this.springs.push(sp);
                this.particles[this.N * i + j].springs.push(sp);
                this.particles[this.N * i + j + this.N].springs.push(sp);
            }
        }
        // 3. Shear 좌상우하
        index = 0;
        for (let i = 0; i < (this.N) * (this.M - 1); i++) {
            if (i % this.N === (this.N - 1)) {
                index++;
                continue;
            }
            const sp = new Spring(
                this.particles[index], 
                this.particles[index + this.N + 1], 
                this.kS, 
                this.kD, 
                "shear",
                index,
                index + this.N + 1
            );
            this.springs.push(sp);
            this.particles[index].springs.push(sp);
                this.particles[index + this.N + 1].springs.push(sp);
            index++;
        }
        // 4. Shear 우상좌하
        index = 0;
        for (let i = 0; i < (this.N) * (this.M - 1); i++) {
            if (i % this.N === 0) {
                index++;
                continue;
            }
            const sp = new Spring(
                this.particles[index], 
                this.particles[index + this.N - 1], 
                this.kS, 
                this.kD, 
                "shear",
                index,
                index + this.N - 1
            );
            this.springs.push(sp);
            this.particles[index].springs.push(sp);
            this.particles[index + this.N - 1].springs.push(sp);
            index++;
        }
        // 5. Bending 가로
        index = 0;
        for (let i = 0; i < (this.N) * this.M; i++) {
            if (i % this.N > this.N - 3) {
                index++;
                continue;
            }
            const sp = new Spring(
                this.particles[index], 
                this.particles[index + 2], 
                this.kS, 
                this.kD, 
                "bending",
                index,
                index + 2
            );
            this.springs.push(sp);
            this.particles[index].springs.push(sp);
            this.particles[index + 2].springs.push(sp);
            index++;
        }
        // //6. Bending 세로
        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.M - 3; j++) {
                const sp = new Spring(
                    this.particles[i + (j * this.M)], 
                    this.particles[i + (j + 3) * this.M], 
                    this.kS, 
                    this.kD, 
                    "bending",
                    i + (j * this.M),
                    i + (j + 3) * this.M
                );
                this.springs.push(sp);
                this.particles[i + (j * this.M)].springs.push(sp);
                this.particles[i + (j + 3) * this.M].springs.push(sp);
            }
        }

        for(let i=0;i<this.particles.length;i++){
            let nConnectedSpring = this.particles[i].springs.length;
            this.maxSpringConnected = Math.max(this.maxSpringConnected, nConnectedSpring);
        }
        console.log("maxSpringConnected : #",this.maxSpringConnected);
    }

    createClothBuffers() {
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

        const fixedData = new Uint32Array(this.particles.length);
        this.particles.forEach((particle, i) => {
            fixedData[i] = particle.fixed ? 1 : 0;
        });

        this.fixedBuffer = this.device.createBuffer({
            size: fixedData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // STORAGE로 사용하며 COPY_DST 플래그를 추가
            mappedAtCreation: true,
        });
        new Uint32Array(this.fixedBuffer.getMappedRange()).set(fixedData);
        this.fixedBuffer.unmap();

        this.springIndicies = new Uint32Array(this.springs.length * 2); // 5 elements per spring
        this.springs.forEach((spring, i) => {
            let offset = i * 2;
            this.springIndicies[offset] = spring.index1;
            this.springIndicies[offset + 1] = spring.index2;
        });        

        this.springRenderBuffer = this.device.createBuffer({
            size: this.springIndicies.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.springRenderBuffer.getMappedRange()).set(this.springIndicies);
        this.springRenderBuffer.unmap();

        const springCalcData = new Float32Array(this.springs.length * 5); // 5 elements per spring
        this.springs.forEach((spring, i) => {
            let offset = i * 5;
            springCalcData[offset] = spring.index1;
            springCalcData[offset + 1] = spring.index2;
            springCalcData[offset + 2] = spring.kS;
            springCalcData[offset + 3] = spring.kD;
            springCalcData[offset + 4] = spring.mRestLen;
        });

        // Create the GPU buffer for springs
        this.springCalculationBuffer = this.device.createBuffer({
            size: springCalcData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.springCalculationBuffer.getMappedRange()).set(springCalcData);
        this.springCalculationBuffer.unmap();

        const numParticlesData = new Uint32Array([this.numParticles]);
        this.numParticlesBuffer = this.device.createBuffer({
            size: numParticlesData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.numParticlesBuffer.getMappedRange()).set(numParticlesData);
        this.numParticlesBuffer.unmap();
    }

    createClothComputePipeline(){
        
        const springComputeShaderModule = this.device.createShaderModule({ code: this.springShader.getSpringUpdateShader() });
        
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 1, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 2, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: {
                        type: 'read-only-storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 3, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                }
            ]
        });
        
        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
        this.computeSpringPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout, // Use the created pipeline layout
            compute: {
                module: springComputeShaderModule,
                entryPoint: 'main',
            },
        });

        const numSpringsData = new Uint32Array([this.springs.length]);
        this.numSpringsBuffer = this.device.createBuffer({
            size: numSpringsData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.numSpringsBuffer.getMappedRange()).set(numSpringsData);
        this.numSpringsBuffer.unmap();

        this.computeSpringBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout, // The layout created earlier
            entries: [
                {
                    binding: 0, // Corresponds to the binding in the layout
                    resource: {
                        buffer: this.positionBuffer // The buffer to bind
                    }
                },
                {
                    binding: 1, // Corresponds to the binding in the layout
                    resource: {
                        buffer: this.velocityBuffer // The buffer to bind
                    }
                },
                {
                    binding: 2, // Corresponds to the binding in the layout
                    resource: {
                        buffer: this.springCalculationBuffer // The buffer to bind
                    }
                },
                {
                    binding: 3, // Corresponds to the binding in the layout
                    resource: {
                        buffer: this.numSpringsBuffer // The buffer to bind
                    }
                }
            ]
        });
    }

    createRenderPipeline() {
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
        console.log("create render pipeline success");
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

    createParticlePipeline(){
        const computeShaderModule = this.device.createShaderModule({ code: this.particleShader.getComputeShader() });
    
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
                    binding: 2, // matches @group(0) @binding(1)
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                }
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
                }
                ,
                {
                    binding: 2,
                    resource: {
                        buffer: this.fixedBuffer,
                    },
                }
            ],
        });
    }

    //Compute Shader
    updateSprings(commandEncoder:GPUCommandEncoder){        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeSpringPipeline);
        computePass.setBindGroup(0, this.computeSpringBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.springs.length / 64.0)+1, 1, 1);
        computePass.end();
    }
    updateParticles(commandEncoder:GPUCommandEncoder) {        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0)+1, 1, 1);
        computePass.end();
    }    
    updateSpringInformations(commandEncoder:GPUCommandEncoder){        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeSpringRenderPipeline);
        computePass.setBindGroup(0, this.computeSpringRenderBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.springs.length / 64.0)+1, 1, 1);
        computePass.end();
    }
    renderCloth(commandEncoder:GPUCommandEncoder){        
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer                
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.draw(this.N * this.M); // Draw the cube using the index count

        passEncoder.setPipeline(this.springPipeline);
        passEncoder.setVertexBuffer(0, this.positionBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
        passEncoder.setIndexBuffer(this.springRenderBuffer, 'uint32'); // 인덱스 포맷 수정
        passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        passEncoder.drawIndexed(this.springIndicies.length);

        passEncoder.end();
    }

    makeRenderpassDescriptor(){
        this.renderPassDescriptor = {
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
    }

    async render() {
        const currentTime = performance.now();
        this.frameCount++;

        this.setCamera(this.camera);
        this.makeRenderpassDescriptor();
        
        const commandEncoder = this.device.createCommandEncoder();
        
        //compute pass
        this.updateSprings(commandEncoder);
        this.updateParticles(commandEncoder);

        //render pass
        this.renderCloth(commandEncoder);
        
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
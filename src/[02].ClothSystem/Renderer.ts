import { vec3 } from 'gl-matrix';
import { ParticleShader } from './ParticleShader';
import { SpringShader } from './SpringShader';
import { RendererOrigin } from '../RendererOrigin';
import { Node, Spring } from '../Physics/Physics';
import { makeFloat32ArrayBuffer, makeFloat32ArrayBufferStorage, makeUInt32ArrayBuffer } from '../WebGPU/Buffer';

export class ClothRenderer extends RendererOrigin {

    private particlePipeline!: GPURenderPipeline;
    private springPipeline!: GPURenderPipeline;
    private trianglePipeline!: GPURenderPipeline;
    private triangleBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private computeSpringPipeline!: GPUComputePipeline;
    private computeSpringBindGroup!: GPUBindGroup;
    private computeSpringRenderPipeline!: GPUComputePipeline;
    private computeSpringRenderBindGroup!: GPUBindGroup;
    private computeNodeForcePipeline!: GPUComputePipeline;
    private computeNodeForceBindGroup!: GPUBindGroup;

    private computeNodeForceInitPipeline!: GPUComputePipeline;
    private computeNodeForceInitBindGroup!: GPUBindGroup;

    //uniform buffers
    private numParticlesBuffer!: GPUBuffer;
    private numSpringsBuffer!: GPUBuffer;
    private maxConnectedSpringBuffer!: GPUBuffer;

    //particle buffers
    private positionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private forceBuffer!: GPUBuffer;
    private fixedBuffer!: GPUBuffer;
    private vertexNormalBuffer!: GPUBuffer;

    //spring buffers
    private springRenderBuffer!: GPUBuffer;
    private springCalculationBuffer!: GPUBuffer;
    private triangleRenderBuffer!: GPUBuffer;

    //shader
    private particleShader!: ParticleShader;
    private springShader!: SpringShader;

    //particle information
    private particles: Node[] = [];
    private uvIndicies: [number, number][] = [];
    private springs: Spring[] = [];
    private springIndicies!: Uint32Array;
    private triangleIndicies!: Uint32Array;
    private mesh!: Float32Array;
    private uv!: Float32Array;
    private meshBuffer!: GPUBuffer;
    private uvBuffer!: GPUBuffer;
    private normals!: Array<vec3>;

    //mateirals
    private texture!: GPUTexture
    private view!: GPUTextureView
    private sampler!: GPUSampler

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
    maxSpringConnected: number = 0;
    private tempSpringForceBuffer!: GPUBuffer;

    constructor(canvasId: string) {
        super(canvasId);
        this.particleShader = new ParticleShader();
        this.springShader = new SpringShader();
    }

    async init() {
        await super.init();
        await this.createAssets();
    }

    createClothModel(x: number, y: number, ks: number, kd: number) {

        this.N = x;
        this.M = y;
        this.kS = ks;
        this.kD = kd;

        this.createParticles();
        this.createSprings();
    }

    calculateNormal(v0: vec3, v1: vec3, v2: vec3) {
        let edge1 = vec3.create();
        let edge2 = vec3.create();
        let normal = vec3.create();

        vec3.subtract(edge1, v1, v0);
        vec3.subtract(edge2, v2, v0);
        vec3.cross(normal, edge1, edge2);
        vec3.normalize(normal, normal);

        return normal;
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

                let u = j / (this.M - 1);
                let v = i / (this.N - 1);

                this.uvIndicies.push([u, v]);
                this.particles.push(n);
            }
        }

        const combinedVertices: number[] = [];
        this.particles.forEach((particle, index) => {
            combinedVertices.push(...particle.position, ...this.uvIndicies[index]);
        });

        // Float32Array로 변환
        const uvs: number[] = [];
        this.uvIndicies.forEach((uv, index) => {
            uvs.push(...uv);
        });
        this.mesh = new Float32Array(combinedVertices);
        this.uv = new Float32Array(uvs);
        let indices: number[] = [];

        this.normals = new Array(this.particles.length);
        this.normals.fill(vec3.create());

        for (let i = 0; i < this.N - 1; i++) {
            for (let j = 0; j < this.M - 1; j++) {
                const topLeft = i * this.M + j;
                const topRight = topLeft + 1;
                const bottomLeft = (i + 1) * this.M + j;
                const bottomRight = bottomLeft + 1;

                indices.push(topLeft, bottomLeft, topRight);
                indices.push(topRight, bottomLeft, bottomRight);

                // 트라이앵글 정점 위치
                let v0 = this.particles[topLeft].position;
                let v1 = this.particles[bottomLeft].position;
                let v2 = this.particles[topRight].position;
                let v3 = this.particles[bottomRight].position;

                // 트라이앵글 표면 노멀 계산
                let triangleNormal1 = this.calculateNormal(v0, v1, v2);
                let triangleNormal2 = this.calculateNormal(v2, v1, v3);

                // 각 정점의 노멀에 트라이앵글의 노멀 누적
                vec3.add(this.normals[topLeft], this.normals[topLeft], triangleNormal1);
                vec3.add(this.normals[bottomLeft], this.normals[bottomLeft], triangleNormal1);
                vec3.add(this.normals[topRight], this.normals[topRight], triangleNormal1);

                vec3.add(this.normals[topRight], this.normals[topRight], triangleNormal2);
                vec3.add(this.normals[bottomLeft], this.normals[bottomLeft], triangleNormal2);
                vec3.add(this.normals[bottomRight], this.normals[bottomRight], triangleNormal2);
            }
        }
        this.normals.forEach(normal => {
            vec3.normalize(normal, normal);
        });

        this.triangleIndicies = new Uint32Array(indices);

        for (let i = 0; i < this.N; i += 2) {
            this.particles[i].fixed = true;
        }

        this.numParticles = this.particles.length;
        console.log("create node success");
    }
    createSprings() {
        let index = 0;
        for (let i = 0; i < this.M; i++) {
            for (let j = 0; j < this.N - 1; j++) {
                if (i > 0 && j === 0) index++;
                const sp = new Spring(
                    this.particles[index],
                    this.particles[index + 1],
                    this.kS,
                    this.kD,
                    "structural",
                    index,
                    index + 1
                );
                sp.targetIndex1 = this.particles[sp.index1].springs.length;
                sp.targetIndex2 = this.particles[sp.index2].springs.length;
                this.springs.push(sp);
                this.particles[sp.index1].springs.push(sp);
                this.particles[sp.index2].springs.push(sp);
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
                sp.targetIndex1 = this.particles[sp.index1].springs.length;
                sp.targetIndex2 = this.particles[sp.index2].springs.length;
                this.springs.push(sp);
                this.particles[sp.index1].springs.push(sp);
                this.particles[sp.index2].springs.push(sp);
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
            sp.targetIndex1 = this.particles[sp.index1].springs.length;
            sp.targetIndex2 = this.particles[sp.index2].springs.length;
            this.springs.push(sp);
            this.particles[sp.index1].springs.push(sp);
            this.particles[sp.index2].springs.push(sp);
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
            sp.targetIndex1 = this.particles[sp.index1].springs.length;
            sp.targetIndex2 = this.particles[sp.index2].springs.length;
            this.springs.push(sp);
            this.particles[sp.index1].springs.push(sp);
            this.particles[sp.index2].springs.push(sp);
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
            sp.targetIndex1 = this.particles[sp.index1].springs.length;
            sp.targetIndex2 = this.particles[sp.index2].springs.length;
            this.springs.push(sp);
            this.particles[sp.index1].springs.push(sp);
            this.particles[sp.index2].springs.push(sp);
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
                sp.targetIndex1 = this.particles[sp.index1].springs.length;
                sp.targetIndex2 = this.particles[sp.index2].springs.length;
                this.springs.push(sp);
                this.particles[sp.index1].springs.push(sp);
                this.particles[sp.index2].springs.push(sp);
            }
        }

        for (let i = 0; i < this.particles.length; i++) {
            let nConnectedSpring = this.particles[i].springs.length;
            this.maxSpringConnected = Math.max(this.maxSpringConnected, nConnectedSpring);
        }
        for (let i = 0; i < this.springs.length; i++) {
            var sp = this.springs[i];

            sp.targetIndex1 += (this.maxSpringConnected * sp.index1);
            sp.targetIndex2 += (this.maxSpringConnected * sp.index2);

            // console.log(i, " => ", sp.index1 , " / ", this.springs[i].targetIndex1);
            // console.log(i, " => ", sp.index2 , " / ", this.springs[i].targetIndex2);
        }
        console.log("maxSpringConnected : #", this.maxSpringConnected);
    }

    async createTextureFromImage(src: string) {
        const response: Response = await fetch(src);
        const blob: Blob = await response.blob();
        const imageData: ImageBitmap = await createImageBitmap(blob);

        await this.loadImageBitmap(this.device, imageData);

        const viewDescriptor: GPUTextureViewDescriptor = {
            format: "rgba8unorm",
            dimension: "2d",
            aspect: "all",
            baseMipLevel: 0,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1
        };
        this.view = this.texture.createView(viewDescriptor);

        const samplerDescriptor: GPUSamplerDescriptor = {
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "nearest",
            mipmapFilter: "nearest",
            maxAnisotropy: 1
        };
        this.sampler = this.device.createSampler(samplerDescriptor);
    }

    async loadImageBitmap(device: GPUDevice, imageData: ImageBitmap) {

        const textureDescriptor: GPUTextureDescriptor = {
            size: {
                width: imageData.width,
                height: imageData.height
            },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        };

        this.texture = device.createTexture(textureDescriptor);

        device.queue.copyExternalImageToTexture(
            { source: imageData },
            { texture: this.texture },
            textureDescriptor.size
        );
    }

    async createAssets() {
        await this.createTextureFromImage("./logo.jpg");
    }

    createClothBuffers() {
        const positionData = new Float32Array(this.particles.flatMap(p => [p.position[0], p.position[1], p.position[2]]));
        const velocityData = new Float32Array(this.particles.flatMap(p => [p.velocity[0], p.velocity[1], p.velocity[2]]));
        const forceData = new Float32Array(this.particles.flatMap(p => [0, 0, 30]));
        const normalData = new Float32Array(this.normals.flatMap(p => [p[0], p[1], p[2]]));

        this.positionBuffer = makeFloat32ArrayBufferStorage(this.device, positionData);
        this.velocityBuffer = makeFloat32ArrayBufferStorage(this.device, velocityData);
        this.forceBuffer = makeFloat32ArrayBufferStorage(this.device, forceData);
        this.vertexNormalBuffer = makeFloat32ArrayBufferStorage(this.device, normalData);

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

        this.springIndicies = new Uint32Array(this.springs.length * 2);
        this.springs.forEach((spring, i) => {
            let offset = i * 2;
            this.springIndicies[offset] = spring.index1;
            this.springIndicies[offset + 1] = spring.index2;
        });

        this.springRenderBuffer = makeUInt32ArrayBuffer(this.device, this.springIndicies);
        this.uvBuffer = makeFloat32ArrayBuffer(this.device, this.mesh);
        this.uvBuffer = makeFloat32ArrayBuffer(this.device, this.uv);

        this.triangleRenderBuffer = this.device.createBuffer({
            size: this.triangleIndicies.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.triangleRenderBuffer.getMappedRange()).set(this.triangleIndicies);
        this.triangleRenderBuffer.unmap();

        const springCalcData = new Float32Array(this.springs.length * 7); // 7 elements per spring
        this.springs.forEach((spring, i) => {
            let offset = i * 7;
            springCalcData[offset] = spring.index1;
            springCalcData[offset + 1] = spring.index2;
            springCalcData[offset + 2] = spring.kS;
            springCalcData[offset + 3] = spring.kD;
            springCalcData[offset + 4] = spring.mRestLen;
            springCalcData[offset + 5] = spring.targetIndex1;
            springCalcData[offset + 6] = spring.targetIndex2;
        });

        // Create the GPU buffer for springs
        this.springCalculationBuffer = this.device.createBuffer({
            size: springCalcData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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

        const nodeSpringConnectedData = new Float32Array(this.maxSpringConnected * this.numParticles * 3);
        this.tempSpringForceBuffer = this.device.createBuffer({
            size: nodeSpringConnectedData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(this.tempSpringForceBuffer.getMappedRange()).set(nodeSpringConnectedData);
        this.tempSpringForceBuffer.unmap();
    }

    createSpringForceComputePipeline() {

        const springComputeShaderModule = this.device.createShaderModule({ code: this.springShader.getSpringUpdateShader() });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0, },
                },
                {
                    binding: 1, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0, },
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
                },
                {
                    binding: 4, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0, },
                },
                {
                    binding: 5, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                }
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.computeSpringPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
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
                    binding: 0,
                    resource: { buffer: this.positionBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.velocityBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.springCalculationBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.numSpringsBuffer }
                },
                {
                    binding: 4,
                    resource: { buffer: this.tempSpringForceBuffer }
                },
                {
                    binding: 5,
                    resource: { buffer: this.numParticlesBuffer }
                }
            ]
        });
    }

    createNodeForceSummationPipeline() {
        const nodeForceComputeShaderModule = this.device.createShaderModule({ code: this.springShader.getNodeForceShader() });
        {
            const bindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'storage', minBindingSize: 0, },
                    },
                    {
                        binding: 1, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'storage', minBindingSize: 0, },
                    },
                    {
                        binding: 2, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                    },
                    {
                        binding: 3, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                    }
                ]
            });

            const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            this.computeNodeForcePipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: nodeForceComputeShaderModule,
                    entryPoint: 'main',
                },
            });

            const maxConnectedSpringData = new Uint32Array([this.maxSpringConnected]);
            this.maxConnectedSpringBuffer = this.device.createBuffer({
                size: maxConnectedSpringData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Uint32Array(this.maxConnectedSpringBuffer.getMappedRange()).set(maxConnectedSpringData);
            this.maxConnectedSpringBuffer.unmap();

            this.computeNodeForceBindGroup = this.device.createBindGroup({
                layout: bindGroupLayout, // The layout created earlier
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.tempSpringForceBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.forceBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.maxConnectedSpringBuffer
                        }
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.numParticlesBuffer
                        }
                    }
                ]
            });
        }
        {
            const bindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'storage', minBindingSize: 0, },
                    },
                    {
                        binding: 1, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'storage', minBindingSize: 0, },
                    },
                    {
                        binding: 2, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                    },
                    {
                        binding: 3, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                    }
                ]
            });

            const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            this.computeNodeForceInitPipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: nodeForceComputeShaderModule,
                    entryPoint: 'initialize',
                },
            });

            const maxConnectedSpringData = new Uint32Array([this.maxSpringConnected]);
            this.maxConnectedSpringBuffer = this.device.createBuffer({
                size: maxConnectedSpringData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Uint32Array(this.maxConnectedSpringBuffer.getMappedRange()).set(maxConnectedSpringData);
            this.maxConnectedSpringBuffer.unmap();

            this.computeNodeForceInitBindGroup = this.device.createBindGroup({
                layout: bindGroupLayout, // The layout created earlier
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.tempSpringForceBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.forceBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.maxConnectedSpringBuffer
                        }
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.numParticlesBuffer
                        }
                    }
                ]
            });
        }
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
                    binding: 0,
                    resource: {
                        buffer: this.mvpUniformBuffer
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

    createTrianglePipeline() {
        const textureShaderModule = this.device.createShaderModule({ code: this.particleShader.getTextureShader() });

        // Assuming bindGroupLayout and pipelineLayout are similar to createParticlePipeline
        // You may reuse the same layout if it fits your needs

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {}
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
            ]
        });

        this.triangleBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.mvpUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: this.view
                },
                {
                    binding: 2,
                    resource: this.sampler
                }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout], // Include the bind group layout created above
        });

        this.trianglePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout, // Reuse or create as needed
            vertex: {
                module: textureShaderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 12,
                    attributes: [
                        {
                            shaderLocation: 0,
                            format: "float32x3",
                            offset: 0
                        }
                    ]
                },
                {
                    arrayStride: 8,
                    attributes: [
                        {
                            shaderLocation: 1,
                            format: "float32x2",
                            offset: 0
                        }
                    ]
                },
                {
                    arrayStride: 12,
                    attributes: [
                        {
                            shaderLocation: 2,
                            format: "float32x2",
                            offset: 0
                        }
                    ]
                }
                ],
            },
            fragment: {
                module: textureShaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'triangle-list',
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

    createParticlePipeline() {
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
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 3,
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
            layout: computePipelineLayout,
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
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.fixedBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.forceBuffer,
                    },
                }
            ],
        });
    }

    //Compute Shader
    updateSprings(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeSpringPipeline);
        computePass.setBindGroup(0, this.computeSpringBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.springs.length / 256.0) + 1, 1, 1);
        computePass.end();
    }
    InitNodeForce(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeNodeForceInitPipeline);
        computePass.setBindGroup(0, this.computeNodeForceInitBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0) + 1, 1, 1);
        computePass.end();
    }
    summationNodeForce(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeNodeForcePipeline);
        computePass.setBindGroup(0, this.computeNodeForceBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0) + 1, 1, 1);
        computePass.end();
    }
    updateParticles(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0) + 1, 1, 1);
        computePass.end();
    }
    updateSpringInformations(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeSpringRenderPipeline);
        computePass.setBindGroup(0, this.computeSpringRenderBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.springs.length / 256.0) + 1, 1, 1);
        computePass.end();
    }
    renderCloth(commandEncoder: GPUCommandEncoder) {
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        // passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        // passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer                
        // passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        // passEncoder.draw(this.N * this.M); // Draw the cube using the index count

        // passEncoder.setPipeline(this.springPipeline);
        // passEncoder.setVertexBuffer(0, this.positionBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
        // passEncoder.setIndexBuffer(this.springRenderBuffer, 'uint32'); // 인덱스 포맷 수정
        // passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        // passEncoder.drawIndexed(this.springIndicies.length);

        passEncoder.setPipeline(this.trianglePipeline);
        passEncoder.setVertexBuffer(0, this.positionBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
        passEncoder.setVertexBuffer(1, this.uvBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
        passEncoder.setVertexBuffer(2, this.vertexNormalBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
        passEncoder.setIndexBuffer(this.triangleRenderBuffer, 'uint32'); // 인덱스 포맷 수정
        passEncoder.setBindGroup(0, this.triangleBindGroup); // Set the bind group with MVP matrix
        passEncoder.drawIndexed(this.triangleIndicies.length);

        passEncoder.end();
    }


    makeRenderpassDescriptor() {
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

    async readBackPositionBuffer() {
        // Create a GPUBuffer for reading back the data
        const readBackBuffer = this.device.createBuffer({
            size: this.forceBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Create a command encoder and copy the position buffer to the readback buffer
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.forceBuffer, 0, readBackBuffer, 0, this.forceBuffer.size);

        // Submit the command to the GPU queue
        const commands = commandEncoder.finish();
        this.device.queue.submit([commands]);

        // Map the readback buffer for reading and read its contents
        await readBackBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBackBuffer.getMappedRange(0, this.forceBuffer.size);
        const data = new Float32Array(arrayBuffer);
        console.log("----");
        for (let i = 0; i < data.length; i += 3) {
            console.log('vec3 Array:', [data[i], data[i + 1], data[i + 2]]);
        }

        // Cleanup
        readBackBuffer.unmap();
        readBackBuffer.destroy();
    }

    async render() {
        const currentTime = performance.now();
        this.frameCount++;
        this.localFrameCount++;

        this.setCamera(this.camera);
        this.makeRenderpassDescriptor();

        const commandEncoder = this.device.createCommandEncoder();

        //compute pass
        this.InitNodeForce(commandEncoder);
        this.updateSprings(commandEncoder);
        this.summationNodeForce(commandEncoder);
        // if(this.localFrameCount%50===0){
        //     await this.readBackPositionBuffer();
        // }

        this.updateParticles(commandEncoder);

        //render pass
        this.renderCloth(commandEncoder);

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        var ms = (currentTime - this.lastTime).toFixed(2);

        if (this.fpsDisplay) {
            this.fpsDisplay.textContent = `${ms}ms`;
        } else {
            console.log(`${ms}ms`);
        }
        this.lastTime = currentTime;
    }
}
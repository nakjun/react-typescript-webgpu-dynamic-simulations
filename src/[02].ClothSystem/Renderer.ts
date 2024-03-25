import { vec3 } from 'gl-matrix';
import { ParticleShader } from './ParticleShader';
import { SpringShader } from './SpringShader';
import { RendererOrigin } from '../RendererOrigin';
import { Node, Spring, Triangle } from '../Physics/Physics';
import { makeFloat32ArrayBuffer, makeFloat32ArrayBufferStorage, makeUInt32IndexArrayBuffer } from '../WebGPU/Buffer';

import { Model } from '../Common/Model';
import { NormalShader } from './NormalShader';

import { ObjLoader, ObjModel } from '../Common/ObjLoader';
import { IntersectionShader } from './IntersectionShader';
import { ObjectShader } from './ObjectShader';

export class ClothRenderer extends RendererOrigin {

    private particlePipeline!: GPURenderPipeline;
    private springPipeline!: GPURenderPipeline;
    private trianglePipeline!: GPURenderPipeline;
    private triangleBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;

    private objRenderPipeline!: GPURenderPipeline;
    private objRenderBindGroup!: GPUBindGroup;

    private computePipeline!: GPUComputePipeline;
    private computeBindGroup!: GPUBindGroup;
    private computeSpringPipeline!: GPUComputePipeline;
    private computeSpringBindGroup!: GPUBindGroup;
    private computeSpringRenderPipeline!: GPUComputePipeline;
    private computeSpringRenderBindGroup!: GPUBindGroup;
    private computeNodeForcePipeline!: GPUComputePipeline;
    private computeNodeForceBindGroup!: GPUBindGroup;
    private computeNormalPipeline!: GPUComputePipeline;
    private computeNormalBindGroup!: GPUBindGroup;
    private computeNormalSummationPipeline!: GPUComputePipeline;
    private computeNormalSummationBindGroup!: GPUBindGroup;

    private computeNodeForceInitPipeline!: GPUComputePipeline;

    private computeIntersectionPipeline!: GPUComputePipeline;
    private computeIntersectionBindGroup!: GPUBindGroup;
    private computeIntersectionBindGroup2!: GPUBindGroup;

    private computeObjectMovePipeline!: GPUComputePipeline;
    private computeObjectMoveBindGroup!: GPUBindGroup;

    private computeIntersectionSummationPipeline!: GPUComputePipeline;

    //uniform buffers
    private numParticlesBuffer!: GPUBuffer;
    private numSpringsBuffer!: GPUBuffer;
    private numTriangleBuffer!: GPUBuffer;
    private maxConnectedTriangleBuffer!: GPUBuffer;

    //particle buffers
    private positionBuffer!: GPUBuffer;
    private prevPositionBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private forceBuffer!: GPUBuffer;
    private fixedBuffer!: GPUBuffer;
    private vertexNormalBuffer!: GPUBuffer;
    private externalForceBuffer!: GPUBuffer;

    //spring buffers
    private springRenderBuffer!: GPUBuffer;
    private springCalculationBuffer!: GPUBuffer;
    private triangleRenderBuffer!: GPUBuffer;
    private triangleCalculationBuffer!: GPUBuffer;

    //shader
    private particleShader!: ParticleShader;
    private springShader!: SpringShader;
    private normalShader!: NormalShader;
    private interesectionShader!: IntersectionShader;
    private objectShader!: ObjectShader;

    //particle information
    private particles: Node[] = [];
    private uvIndices: [number, number][] = [];
    private triangles: Triangle[] = [];
    private springs: Spring[] = [];
    private springIndices!: Uint32Array;
    private triangleIndices!: Uint32Array;
    private uv!: Float32Array;
    private uvBuffer!: GPUBuffer;
    private normals!: Array<vec3>;

    //mateirals - cloth
    private texture!: GPUTexture
    private view!: GPUTextureView
    private sampler!: GPUSampler
    private camPosBuffer!: GPUBuffer;
    private lightDataBuffer!: GPUBuffer;
    private alphaValueBuffer!: GPUBuffer;


    numParticles: number = 0;

    renderPassDescriptor!: GPURenderPassDescriptor;

    //cloth information
    N: number = 0;
    M: number = 0;

    structuralKs: number = 1000.0;
    shearKs: number = 1000.0;
    bendKs: number = 1000.0;

    kD: number = 0;

    xSize: number = 300.0;
    ySize: number = 300.0;

    //for temp storage buffer
    maxSpringConnected: number = 0;
    private tempSpringForceBuffer!: GPUBuffer;

    maxTriangleConnected: number = 0;
    private tempTriangleNormalBuffer!: GPUBuffer;

    private tempTriangleTriangleCollisionBuffer!: GPUBuffer;

    //sphere model
    private modelGenerator!: Model;
    private sphereRadious!: number;
    private sphereSegments!: number;
    private spherePosition!: vec3;
    private ObjectPosBuffer!: GPUBuffer;
    private objectIndexBuffer!: GPUBuffer;
    private objectUVBuffer!: GPUBuffer;
    private objectNormalBuffer!: GPUBuffer;
    private objectIndicesLength!: number;
    private objectBindGroup!: GPUBindGroup;
    private objectNumTriangleBuffer!: GPUBuffer;

    //mateirals - sphere
    private textureObject!: GPUTexture
    private viewObject!: GPUTextureView
    private samplerObject!: GPUSampler

    //3d objects
    private model!: ObjModel;

    //collision response
    private collisionTempBuffer!: GPUBuffer;
    private collisionCountTempBuffer!: GPUBuffer;

    /* constructor */
    constructor(canvasId: string) {
        super(canvasId);
        this.particleShader = new ParticleShader();
        this.springShader = new SpringShader();
        this.normalShader = new NormalShader();
        this.interesectionShader = new IntersectionShader();
        this.objectShader = new ObjectShader();

        this.modelGenerator = new Model();
        this.model = new ObjModel();

        console.log("hhh")
    }

    /* async functions */
    async init() {
        await super.init();
        await this.createAssets();
        //await this.MakeModelData();
        await this.createSphereModel();
    }
    async createTextureFromImage(src: string, device: GPUDevice): Promise<{ texture: GPUTexture, sampler: GPUSampler, view: GPUTextureView }> {
        const response: Response = await fetch(src);
        const blob: Blob = await response.blob();
        const imageData: ImageBitmap = await createImageBitmap(blob);

        const texture = await this.loadImageBitmap(device, imageData);

        const view = texture.createView({
            format: "rgba8unorm",
            dimension: "2d",
            aspect: "all",
            baseMipLevel: 0,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1
        });

        const sampler = device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "nearest",
            mipmapFilter: "nearest",
            maxAnisotropy: 1
        });

        return { texture, sampler, view };
    }
    async loadImageBitmap(device: GPUDevice, imageData: ImageBitmap): Promise<GPUTexture> {

        const textureDescriptor: GPUTextureDescriptor = {
            size: {
                width: imageData.width,
                height: imageData.height
            },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        };

        const texture = device.createTexture(textureDescriptor);

        device.queue.copyExternalImageToTexture(
            { source: imageData },
            { texture: texture },
            { width: imageData.width, height: imageData.height },
        );

        return texture;
    }
    async createAssets() {
        const assets1 = await this.createTextureFromImage("./textures/siggraph.png", this.device);
        this.texture = assets1.texture;
        this.sampler = assets1.sampler;
        this.view = assets1.view;

        const assets2 = await this.createTextureFromImage("./textures/metal.jpg", this.device);
        this.textureObject = assets2.texture;
        this.samplerObject = assets2.sampler;
        this.viewObject = assets2.view;
    }
    async readBackPositionBuffer() {
        var target = this.collisionTempBuffer;

        // Create a GPUBuffer for reading back the data
        const readBackBuffer = this.device.createBuffer({
            size: target.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Create a command encoder and copy the position buffer to the readback buffer
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(target, 0, readBackBuffer, 0, target.size);

        // Submit the command to the GPU queue
        const commands = commandEncoder.finish();
        this.device.queue.submit([commands]);

        // Map the readback buffer for reading and read its contents
        await readBackBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBackBuffer.getMappedRange(0, target.size);
        const data = new Int32Array(arrayBuffer);
        // console.log("----");
        for (let i = 0; i < data.length; i += 3) {
            if (data[i] === 0 && data[i + 1] === 0 && data[i + 2] === 0) { continue; }
            console.log('[', i / 3, ']vec3 Array:', [data[i], data[i + 1], data[i + 2]]);
        }

        // for (let i = 0; i < data.length; i += 1) {            
        //     //if(data[i]===0 && data[i+1]===0 && data[i+2]===0){continue;}
        //     console.log('[', i, '] Array:', data[i]);
        // }

        // var res = JSON.stringify(data, undefined);
        // console.log(res);

        // Cleanup
        readBackBuffer.unmap();
        readBackBuffer.destroy();
    }
    async render() {
        const currentTime = performance.now();

        this.setCamera(this.camera);
        this.makeRenderpassDescriptor();

        const commandEncoder = this.device.createCommandEncoder();

        if (this.renderOptions.wind) {
            const newExternalForce = new Float32Array([0.0, 0.0, 20.0]);
            this.device.queue.writeBuffer(
                this.externalForceBuffer,
                0, // Buffer 내에서의 시작 위치
                newExternalForce.buffer, // 새로운 데이터
                newExternalForce.byteOffset,
                newExternalForce.byteLength
            );
        }

        //compute pass
        this.updateObjects(commandEncoder);
        this.InitNodeForce(commandEncoder);
        this.updateSprings(commandEncoder);
        this.summationNodeForce(commandEncoder);
        //this.Intersections(commandEncoder);


        this.updateParticles(commandEncoder);
        this.updateNormals(commandEncoder);
        //render pass
        this.renderCloth(commandEncoder);

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        this.stats.ms = (currentTime - this.lastTime).toFixed(2);
        this.stats.fps = Math.round(1000.0 / (currentTime - this.lastTime));

        this.lastTime = currentTime;
        this.localFrameCount++;
    }

    /* Create Data */
    createSphereModel() {
        this.sphereRadious = 4.0;
        this.sphereSegments = 64;
        this.spherePosition = vec3.fromValues(30.0, 30.0, -10.0);
        var sphere = this.modelGenerator.createSphere(this.sphereRadious, this.sphereSegments, this.spherePosition);

        var sphere2 = this.modelGenerator.createSphere(this.sphereRadious, this.sphereSegments, vec3.fromValues(327.0
            , 30.0, -10.0));

        var vertArray = new Float32Array([...sphere.vertices, ...sphere2.vertices]);
        //var indArray = new Uint32Array(sphere.indices);
        var uvArray = new Float32Array(sphere.uvs);
        var normalArray = new Float32Array(sphere.normals);
        this.objectIndicesLength = sphere.indices.length;

            // 인덱스 오프셋 적용
        var vertexOffset = sphere.vertices.length / 3; // 각 정점은 3개의 값(x, y, z)으로 구성됨
        var sphere2IndicesWithOffset = sphere2.indices.map(index => index + vertexOffset);
        var combinedIndArray = new Uint32Array([...sphere.indices, ...sphere2IndicesWithOffset]);

        this.objectIndicesLength = combinedIndArray.length;

        console.log("this object's indices length: " + this.objectIndicesLength / 3);

        this.ObjectPosBuffer = makeFloat32ArrayBufferStorage(this.device, vertArray);
        this.objectIndexBuffer = makeUInt32IndexArrayBuffer(this.device, combinedIndArray);
        this.objectUVBuffer = makeFloat32ArrayBufferStorage(this.device, uvArray);
        this.objectNormalBuffer = makeFloat32ArrayBufferStorage(this.device, normalArray);

        const numTriangleData = new Uint32Array([sphere.indices.length / 3]);
        this.objectNumTriangleBuffer = this.device.createBuffer({
            size: numTriangleData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.objectNumTriangleBuffer.getMappedRange()).set(numTriangleData);
        this.objectNumTriangleBuffer.unmap();

        const shaderModule = this.device.createShaderModule({ code: this.objectShader.getMoveShader() });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0, },
                },
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        this.computeObjectMovePipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        this.computeObjectMoveBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout, // The layout created earlier
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.ObjectPosBuffer }
                },
            ]
        });



        const materialShaderModule = this.device.createShaderModule({ code: this.objectShader.getMaterialShader() });
        const bindGroupLayout2 = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
            ]
        });

        this.camPosBuffer = this.device.createBuffer({
            size: 4 * Float32Array.BYTES_PER_ELEMENT, // vec3<f32> + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.lightDataBuffer = this.device.createBuffer({
            size: 48, // vec3 position (12 bytes) + padding (4 bytes) + vec4 color (16 bytes) + intensity (4 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.mvpUniformBuffer = this.device.createBuffer({
            size: 64 * 3, // The total size needed for the matrices
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST // The buffer is used as a uniform and can be copied to
        });

        this.objRenderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout2,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.mvpUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.camPosBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.lightDataBuffer
                    }
                }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout2],
        });

        this.objRenderPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: materialShaderModule,
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
                            format: "float32x3",
                            offset: 0
                        }
                    ]
                }
                ],
            },
            fragment: {
                module: materialShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format, blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
            multisample: {
                count: this.sampleCount,
            },
        });
    }

    async MakeModelData() {
        const loader = new ObjLoader();

        //this.model = await loader.load('./objects/skybox.obj', 10.0);
        //this.model = await loader.load('./objects/bunny.obj', 100.0);
        //this.model = await loader.load('./objects/armadillo4.obj', 30.0);
        this.model = await loader.load('./objects/dragon2.obj', 2.0);

        console.log("object file load end");

        var vertArray = new Float32Array(this.model.vertices);
        var indArray = new Uint32Array(this.model.indices);
        var normalArray = new Float32Array(this.model.normals);
        var uvArray = new Float32Array(this.model.uvs);
        this.objectIndicesLength = this.model.indices.length;

        console.log("this object's indices length: " + this.objectIndicesLength / 3);

        this.ObjectPosBuffer = makeFloat32ArrayBufferStorage(this.device, vertArray);
        this.objectIndexBuffer = makeUInt32IndexArrayBuffer(this.device, indArray);
        this.objectUVBuffer = makeFloat32ArrayBufferStorage(this.device, uvArray);
        this.objectNormalBuffer = makeFloat32ArrayBufferStorage(this.device, normalArray);

        const numTriangleData = new Uint32Array([this.model.indices.length / 3]);
        this.objectNumTriangleBuffer = this.device.createBuffer({
            size: numTriangleData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.objectNumTriangleBuffer.getMappedRange()).set(numTriangleData);
        this.objectNumTriangleBuffer.unmap();

        const shaderModule = this.device.createShaderModule({ code: this.objectShader.getMoveShader() });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0, },
                },
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        this.computeObjectMovePipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        this.computeObjectMoveBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout, // The layout created earlier
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.ObjectPosBuffer }
                },
            ]
        });



        const materialShaderModule = this.device.createShaderModule({ code: this.objectShader.getMaterialShader() });
        const bindGroupLayout2 = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
            ]
        });

        this.camPosBuffer = this.device.createBuffer({
            size: 4 * Float32Array.BYTES_PER_ELEMENT, // vec3<f32> + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.lightDataBuffer = this.device.createBuffer({
            size: 48, // vec3 position (12 bytes) + padding (4 bytes) + vec4 color (16 bytes) + intensity (4 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.mvpUniformBuffer = this.device.createBuffer({
            size: 64 * 3, // The total size needed for the matrices
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST // The buffer is used as a uniform and can be copied to
        });

        this.objRenderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout2,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.mvpUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.camPosBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.lightDataBuffer
                    }
                }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout2],
        });

        this.objRenderPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: materialShaderModule,
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
                            format: "float32x3",
                            offset: 0
                        }
                    ]
                }
                ],
            },
            fragment: {
                module: materialShaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format, blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
            multisample: {
                count: this.sampleCount,
            },
        });
    }

    createTriTriIntersectionPipeline() {
        const intersectionComputeShaderModule = this.device.createShaderModule({ code: this.interesectionShader.getTriTriIntersectionShader() });

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
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 3, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 4, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4, },
                },
                {
                    binding: 5, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 6, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 7, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 8, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        this.computeIntersectionPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: intersectionComputeShaderModule,
                entryPoint: 'main',
            },
        });

        this.computeIntersectionBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout, // The layout created earlier
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.positionBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.triangleRenderBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.ObjectPosBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.objectIndexBuffer }
                },
                {
                    binding: 4,
                    resource: { buffer: this.numTriangleBuffer }
                },
                {
                    binding: 5,
                    resource: { buffer: this.objectNumTriangleBuffer }
                },
                {
                    binding: 6,
                    resource: { buffer: this.collisionTempBuffer }
                },
                {
                    binding: 7,
                    resource: { buffer: this.collisionCountTempBuffer }
                },
                {
                    binding: 8,
                    resource: { buffer: this.velocityBuffer }
                },
            ]
        });
    }

    createIntersectionPipeline() {
        const intersectionComputeShaderModule = this.device.createShaderModule({ code: this.interesectionShader.getIntersectionShader() });

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
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 3, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 4, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4, },
                },
                {
                    binding: 5, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 6, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 7, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 8, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
                {
                    binding: 9, // The binding number in the shader
                    visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                    buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                },
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.computeIntersectionSummationPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: intersectionComputeShaderModule,
                entryPoint: 'response',
            },
        });

        this.computeIntersectionBindGroup2 = this.device.createBindGroup({
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
                    resource: { buffer: this.ObjectPosBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.objectIndexBuffer }
                },
                {
                    binding: 4,
                    resource: { buffer: this.numParticlesBuffer }
                },
                {
                    binding: 5,
                    resource: { buffer: this.objectNumTriangleBuffer }
                },
                {
                    binding: 6,
                    resource: { buffer: this.collisionTempBuffer }
                },
                {
                    binding: 7,
                    resource: { buffer: this.fixedBuffer }
                },
                {
                    binding: 8,
                    resource: { buffer: this.collisionCountTempBuffer }
                },
                {
                    binding: 9,
                    resource: { buffer: this.prevPositionBuffer }
                },
            ]
        });
    }

    gaussianRandom() {
        let rand = 0;
        for (let i = 0; i < 6; i += 1) {
            rand += Math.random();
        }
        return rand / 6;
    }

    createClothModel(x: number, y: number, structuralKs: number = 5000.0, shearKs: number = 2000.0, bendKs: number = 500.0, kd: number = 0.25) {

        this.N = x;
        this.M = y;
        this.structuralKs = structuralKs;
        this.shearKs = shearKs;
        this.bendKs = bendKs;
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
        //20x20 cloth
        const start_x = 30;
        const start_y = 30;

        // const start_x = 15;
        // const start_y = 10;

        const dist_x = (this.xSize / this.N);
        const dist_y = (this.ySize / this.M);

        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.M; j++) {
                var pos = vec3.fromValues(start_x + (dist_x * j), start_y - (dist_y * i), -10.0);
                //var pos = vec3.fromValues(start_x - (dist_x * j), 100.0, start_y - (dist_y * i));
                var vel = vec3.fromValues(0, 0, 0);

                const n = new Node(pos, vel);

                let u = j / (this.M - 1);
                let v = i / (this.N - 1);

                this.uvIndices.push([u, v]);
                this.particles.push(n);
            }
        }

        // const dist_x = (this.xSize / this.N);
        // const dist_y = (this.ySize / this.M);
        // const maxHeight = 27.0; // 최대 높이 설정
        // const minHeight = 13.0; // 최소 높이 설정

        // // 중심점 위치 계산
        // const centerX = (this.N - 1) / 2;
        // const centerY = (this.M - 1) / 2;

        // for (let i = 0; i < this.N; i++) {
        //     for (let j = 0; j < this.M; j++) {
        //         // 중심으로부터의 거리에 따른 높이 조정
        //         let distanceFromCenter = Math.sqrt(Math.pow(i - centerX, 2) + Math.pow(j - centerY, 2));
        //         let heightFactor = (distanceFromCenter / Math.max(centerX, centerY)) * (maxHeight - minHeight);
        //         let yPos = (maxHeight + heightFactor) - 7.0;

        //         var pos = vec3.fromValues(start_x - (dist_x * j), yPos, start_y - (dist_y * i));
        //         var vel = vec3.fromValues(0, 0, 0);

        //         const n = new Node(pos, vel);

        //         let u = j / (this.M - 1);
        //         let v = i / (this.N - 1);

        //         this.uvIndices.push([u, v]);
        //         this.particles.push(n);
        //     }
        // }
        
        const combinedVertices: number[] = [];
        this.particles.forEach((particle, index) => {
            combinedVertices.push(...particle.position, ...this.uvIndices[index]);
        });

        // Float32Array로 변환
        const uvs: number[] = [];
        this.uvIndices.forEach((uv, index) => {
            uvs.push(...uv);
        });
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

                var triangle1 = new Triangle(topLeft, bottomLeft, topRight);
                this.triangles.push(triangle1);
                this.particles[topLeft].triangles.push(triangle1);
                this.particles[bottomLeft].triangles.push(triangle1);
                this.particles[topRight].triangles.push(triangle1);

                var triangle2 = new Triangle(topRight, bottomLeft, bottomRight);
                this.triangles.push(triangle2);
                this.particles[topRight].triangles.push(triangle2);
                this.particles[bottomLeft].triangles.push(triangle2);
                this.particles[bottomRight].triangles.push(triangle2);

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

        this.triangleIndices = new Uint32Array(indices);

        //first line fix
        // for (let i = 0; i < 5; i++) {
        //     this.particles[i].fixed = true;
        // }

        // for (let i = this.N-6; i < this.N; i++) {
        //     this.particles[i].fixed = true;
        // }

        // for (let i = 0; i < this.N / 3; i++) {
        //     this.particles[i].fixed = true;
        // }
        // for (let i = this.N / 2; i < this.N; i++) {
        //     this.particles[i].fixed = true;
        // }
        
        //0, N fix       
        this.particles[0].fixed = true;
        this.particles[this.N-1].fixed = true;
        console.log(this.particles[this.N-1].position)

        this.numParticles = this.particles.length;
        console.log("make #", this.numParticles, " particles create success");
        console.log("make #", indices.length, " faces create success");
        for (let i = 0; i < this.particles.length; i++) {
            let nConnectedTriangle = this.particles[i].triangles.length;
            this.maxTriangleConnected = Math.max(this.maxTriangleConnected, nConnectedTriangle);
        }
        console.log(this.maxTriangleConnected);
    }
    createSprings() {
        let index = 0;
        for (let i = 0; i < this.M; i++) {
            for (let j = 0; j < this.N - 1; j++) {
                if (i > 0 && j === 0) index++;
                const sp = new Spring(
                    this.particles[index],
                    this.particles[index + 1],
                    this.structuralKs,
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
                    this.structuralKs,
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
                this.shearKs,
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
                this.shearKs,
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
                this.bendKs,
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
                    this.bendKs,
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
        console.log("make #", this.springs.length, " spring create success");
    }
    createClothBuffers() {
        const positionData = new Float32Array(this.particles.flatMap(p => [p.position[0], p.position[1], p.position[2]]));
        const velocityData = new Float32Array(this.particles.flatMap(p => [p.velocity[0], p.velocity[1], p.velocity[2]]));
        const forceData = new Float32Array(this.particles.flatMap(p => [0.0, 0.0, 0.0]));
        const normalData = new Float32Array(this.normals.flatMap(p => [p[0], p[1], p[2]]));

        this.positionBuffer = makeFloat32ArrayBufferStorage(this.device, positionData);
        this.prevPositionBuffer = makeFloat32ArrayBufferStorage(this.device, positionData);
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

        this.springIndices = new Uint32Array(this.springs.length * 2);
        this.springs.forEach((spring, i) => {
            let offset = i * 2;
            this.springIndices[offset] = spring.index1;
            this.springIndices[offset + 1] = spring.index2;
        });

        this.springRenderBuffer = makeUInt32IndexArrayBuffer(this.device, this.springIndices);

        this.uvBuffer = makeFloat32ArrayBuffer(this.device, this.uv);

        this.triangleRenderBuffer = this.device.createBuffer({
            size: this.triangleIndices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Uint32Array(this.triangleRenderBuffer.getMappedRange()).set(this.triangleIndices);
        this.triangleRenderBuffer.unmap();

        const springCalcData = new Float32Array(this.springs.length * 5); // 7 elements per spring
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
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(this.springCalculationBuffer.getMappedRange()).set(springCalcData);
        this.springCalculationBuffer.unmap();

        const triangleCalcData = new Float32Array(this.triangles.length * 3); // 7 elements per spring
        this.triangles.forEach((triangle, i) => {
            let offset = i * 3;
            triangleCalcData[offset] = triangle.v1;
            triangleCalcData[offset + 1] = triangle.v2;
            triangleCalcData[offset + 2] = triangle.v3;
        });
        this.triangleCalculationBuffer = this.device.createBuffer({
            size: triangleCalcData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(this.triangleCalculationBuffer.getMappedRange()).set(triangleCalcData);
        this.triangleCalculationBuffer.unmap();

        const numParticlesData = new Uint32Array([this.numParticles]);
        this.numParticlesBuffer = this.device.createBuffer({
            size: numParticlesData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(this.numParticlesBuffer.getMappedRange()).set(numParticlesData);
        this.numParticlesBuffer.unmap();

        const nodeSpringConnectedData = new Int32Array(this.numParticles * 3);
        this.tempSpringForceBuffer = this.device.createBuffer({
            size: nodeSpringConnectedData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Int32Array(this.tempSpringForceBuffer.getMappedRange()).set(nodeSpringConnectedData);
        this.tempSpringForceBuffer.unmap();

        const nodeTriangleConnectedData = new Uint32Array(this.numParticles * 3);
        this.tempTriangleNormalBuffer = this.device.createBuffer({
            size: nodeTriangleConnectedData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint32Array(this.tempTriangleNormalBuffer.getMappedRange()).set(nodeTriangleConnectedData);
        this.tempTriangleNormalBuffer.unmap();


        const collisionTempData = new Int32Array(this.numParticles * 3);
        this.collisionTempBuffer = this.device.createBuffer({
            size: collisionTempData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Int32Array(this.collisionTempBuffer.getMappedRange()).set(collisionTempData);
        this.collisionTempBuffer.unmap();

        const collisionCountTempData = new Int32Array(this.numParticles);
        this.collisionCountTempBuffer = this.device.createBuffer({
            size: collisionCountTempData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Int32Array(this.collisionCountTempBuffer.getMappedRange()).set(collisionCountTempData);
        this.collisionCountTempBuffer.unmap();
    }
    /* Compute Shader Pipeline */
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
            ]
        });
        {   /*  Node Force Merge Equation */

            const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            this.computeNodeForcePipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: nodeForceComputeShaderModule,
                    entryPoint: 'main',
                },
            });
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
                            buffer: this.numParticlesBuffer
                        }
                    }
                ]
            });
        }
        {
            const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            this.computeNodeForceInitPipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: nodeForceComputeShaderModule,
                    entryPoint: 'initialize',
                },
            });
        }
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
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                        minBindingSize: 0, // or specify the actual size
                    },
                },
                {
                    binding: 5, // This matches @group(0) @binding(5) in the WGSL shader
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform',
                        minBindingSize: 0, // Specify the size of vec3<f32>
                    },
                },
            ],
        });

        const initialExternalForce = new Float32Array([0.0, 0.0, 0.0]);

        // externalForceBuffer 생성
        this.externalForceBuffer = this.device.createBuffer({
            size: initialExternalForce.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.externalForceBuffer.getMappedRange()).set(initialExternalForce);
        this.externalForceBuffer.unmap();

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
                },
                {
                    binding: 4,
                    resource: {
                        buffer: this.prevPositionBuffer,
                    },
                },
                {
                    binding: 5,
                    resource: {
                        buffer: this.externalForceBuffer,
                    },
                },
            ],
        });
    }
    createUpdateNormalPipeline() {
        const normalComputeShaderModule = this.device.createShaderModule({ code: this.normalShader.getNormalUpdateComputeShader() });
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
                        buffer: { type: 'storage', minBindingSize: 0 }, // Ensure this matches the shader's expectation
                    },
                    {
                        binding: 3, // The binding number in the shader
                        visibility: GPUShaderStage.COMPUTE, // Accessible from the vertex shader
                        buffer: { type: 'uniform', minBindingSize: 4 }, // Ensure this matches the shader's expectation
                    }
                ]
            });

            const computePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            this.computeNormalPipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: normalComputeShaderModule,
                    entryPoint: 'main',
                },
            });

            const numTriangleData = new Uint32Array([this.triangles.length]);
            this.numTriangleBuffer = this.device.createBuffer({
                size: numTriangleData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Uint32Array(this.numTriangleBuffer.getMappedRange()).set(numTriangleData);
            this.numTriangleBuffer.unmap();

            this.computeNormalBindGroup = this.device.createBindGroup({
                layout: bindGroupLayout, // The layout created earlier
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.positionBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.triangleCalculationBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.tempTriangleNormalBuffer
                        }
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.numTriangleBuffer
                        }
                    }
                ]
            });
        }
        const normalSummationComputeShaderModule = this.device.createShaderModule({ code: this.normalShader.getNormalSummationComputeShader() });
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

            this.computeNormalSummationPipeline = this.device.createComputePipeline({
                layout: computePipelineLayout,
                compute: {
                    module: normalSummationComputeShaderModule,
                    entryPoint: 'main',
                },
            });

            const maxConnectedTriangleData = new Uint32Array([this.maxTriangleConnected]);
            this.maxConnectedTriangleBuffer = this.device.createBuffer({
                size: maxConnectedTriangleData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Uint32Array(this.maxConnectedTriangleBuffer.getMappedRange()).set(maxConnectedTriangleData);
            this.maxConnectedTriangleBuffer.unmap();

            this.computeNormalSummationBindGroup = this.device.createBindGroup({
                layout: bindGroupLayout, // The layout created earlier
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.tempTriangleNormalBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.vertexNormalBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.maxConnectedTriangleBuffer
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

    /* Render Shader Pipeline */
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
                targets: [{
                    format: this.format, blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
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
            multisample: {
                count: this.sampleCount,
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
                targets: [{
                    format: this.format, blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
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
            multisample: {
                count: this.sampleCount,
            },
        });
    }
    createTrianglePipeline() {
        const textureShaderModule = this.device.createShaderModule({ code: this.particleShader.getTextureShader() });
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
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
            ]
        });

        const alphaData = new Float32Array([1.0]);
        this.alphaValueBuffer = this.device.createBuffer({
            size: alphaData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.alphaValueBuffer.getMappedRange()).set(alphaData);
        this.alphaValueBuffer.unmap();

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
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.camPosBuffer
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: this.lightDataBuffer
                    }
                },
                {
                    binding: 5,
                    resource: {
                        buffer: this.alphaValueBuffer
                    }
                }
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        this.trianglePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
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
                targets: [{
                    format: this.format, blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
                //topology: 'line-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
            multisample: {
                count: this.sampleCount,
            },
        });
    }

    /* Update Routine */
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
        computePass.setBindGroup(0, this.computeNodeForceBindGroup);
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
    updateNormals(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeNormalPipeline);
        computePass.setBindGroup(0, this.computeNormalBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.triangles.length / 256.0) + 1, 1, 1);
        computePass.end();

        const computePass2 = commandEncoder.beginComputePass();
        computePass2.setPipeline(this.computeNormalSummationPipeline);
        computePass2.setBindGroup(0, this.computeNormalSummationBindGroup);
        computePass2.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0) + 1, 1, 1);
        computePass2.end();
    }

    Intersections(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computeIntersectionPipeline);
        computePass.setBindGroup(0, this.computeIntersectionBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.triangleIndices.length / 16.0) + 1, (this.objectIndicesLength / 3) / 16.0 + 1, 1);
        computePass.end();


        if (this.localFrameCount % 50 === 0) {
            this.readBackPositionBuffer();
        }

        const computePass2 = commandEncoder.beginComputePass();
        computePass2.setPipeline(this.computeIntersectionSummationPipeline);
        computePass2.setBindGroup(0, this.computeIntersectionBindGroup2);
        computePass2.dispatchWorkgroups(Math.ceil(this.numParticles / 256.0) + 1, 1, 1);
        computePass2.end();
    }

    updateObjects(commandEncoder: GPUCommandEncoder) {
        if (this.renderOptions.moveObject) {
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.computeObjectMovePipeline);
            computePass.setBindGroup(0, this.computeObjectMoveBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(this.model.vertices.length / 256.0) + 1, 1, 1);
            computePass.end();
        }
    }

    renderCloth(commandEncoder: GPUCommandEncoder) {
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

        this.device.queue.writeBuffer(
            this.camPosBuffer,
            0,
            new Float32Array([...this.camera.position, 1.0]) // vec3 + padding
        );

        let lightData = [this.light_position[0], this.light_position[1], this.light_position[2], 0.0, this.light_color[0], this.light_color[1], this.light_color[2], 1.0, this.light_intensity, this.specular_strength, this.shininess, 0.0];
        this.device.queue.writeBuffer(this.lightDataBuffer, 0, new Float32Array(lightData));

        if (this.renderOptions.renderObject) {
            passEncoder.setPipeline(this.objRenderPipeline);
            passEncoder.setVertexBuffer(0, this.ObjectPosBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setVertexBuffer(1, this.objectUVBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setVertexBuffer(2, this.objectNormalBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setIndexBuffer(this.objectIndexBuffer, 'uint32'); // 인덱스 포맷 수정
            passEncoder.setBindGroup(0, this.objRenderBindGroup); // Set the bind group with MVP matrix
            passEncoder.drawIndexed(this.objectIndicesLength);
        }

        if (this.renderOptions.wireFrame) {
            passEncoder.setPipeline(this.springPipeline);
            passEncoder.setVertexBuffer(0, this.positionBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setIndexBuffer(this.springRenderBuffer, 'uint32'); // 인덱스 포맷 수정
            passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
            passEncoder.drawIndexed(this.springIndices.length);

            passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
            passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer                
            passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
            passEncoder.draw(this.N * this.M); // Draw the cube using the index count
        }
        else {
            passEncoder.setPipeline(this.trianglePipeline);
            passEncoder.setVertexBuffer(0, this.positionBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setVertexBuffer(1, this.uvBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setVertexBuffer(2, this.vertexNormalBuffer); // 정점 버퍼 설정, 스프링의 경우 필요에 따라
            passEncoder.setIndexBuffer(this.triangleRenderBuffer, 'uint32'); // 인덱스 포맷 수정
            passEncoder.setBindGroup(0, this.triangleBindGroup); // Set the bind group with MVP matrix
            passEncoder.drawIndexed(this.triangleIndices.length);
        }

        // if(this.renderOptions.wind){
        //     passEncoder.setPipeline(this.particlePipeline); // Your render pipeline        
        //     passEncoder.setVertexBuffer(0, this.positionBuffer); // Set the vertex buffer                
        //     passEncoder.setBindGroup(0, this.renderBindGroup); // Set the bind group with MVP matrix
        //     passEncoder.draw(this.N * this.M); // Draw the cube using the index count
        // }

        passEncoder.end();
    }

    /* Make Metadata */
    makeRenderpassDescriptor() {
        this.renderPassDescriptor = {
            colorAttachments: [{
                view: this.resolveTexture.createView(),
                resolveTarget: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.25, g: 0.25, b: 0.25, a: 1.0 }, // Background color
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: { // Add this attachment for depth testing
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };
    }
}
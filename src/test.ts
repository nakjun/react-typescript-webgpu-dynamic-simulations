import { Particle } from "./particle";

export const Initialize = async () => {
  const canvas: HTMLCanvasElement = <HTMLCanvasElement>(
    document.getElementById("gfx-main")
  );

  const adapter: GPUAdapter = <GPUAdapter>(
    await navigator.gpu?.requestAdapter()
  );
  const device: GPUDevice = <GPUDevice>await adapter?.requestDevice();
  const context: GPUCanvasContext = <GPUCanvasContext>(
    canvas.getContext("webgpu")
  );
  const format: GPUTextureFormat = "bgra8unorm";
  context.configure({
    device: device,
    format: format,
    alphaMode: "opaque",
  });

  const commandEncoder = device.createCommandEncoder();
  const textureView: GPUTextureView = context.getCurrentTexture().createView();

  const clearColor = { r: 0.25, g: 1.0, b: 0.5, a: 1.0 }; // Gray background
  const colorAttachment: GPURenderPassColorAttachment = {
    view: textureView,
    loadOp: "clear",
    storeOp: "store",
    clearValue: clearColor,
  };

  const renderpass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
    colorAttachments: [colorAttachment],
  });

  const particles = [
    new Particle([0.0, 0.5], [1.0, 0.0, 0.0]),
    new Particle([-0.5, -0.5], [0.0, 1.0, 0.0]),
    new Particle([0.5, -0.5], [0.0, 0.0, 1.0])
  ];

  const shaders = `struct Fragment {
    @builtin(position) Position : vec4<f32>,
    @location(0) Color : vec4<f32>
};

@vertex
fn vs_main(@builtin(vertex_index) v_id: u32) -> Fragment {

    //pre-bake positions and colors, for now.
    var positions = array<vec2<f32>, 3> (
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5)
    );

    var colors = array<vec3<f32>, 3> (
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );

    var output : Fragment;
    output.Position = vec4<f32>(positions[v_id], 0.0, 1.0);
    output.Color = vec4<f32>(colors[v_id], 1.0);

    return output;
}

@fragment
fn fs_main(@location(0) Color: vec4<f32>) -> @location(0) vec4<f32> {
    return Color;
}`;

  const shaderModule = device.createShaderModule({ code: shaders });

  // 적절한 bindGroupLayout 생성
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [],
  });

    // Create an empty bind group
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [],
    });

  // 렌더 파이프라인 생성 시 layout 속성에 bindGroupLayout 할당
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout, // layout 속성 추가
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: format,
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // 삼각형 그리기 명령 추가
  renderpass.setPipeline(pipeline);
  renderpass.setBindGroup(0, bindGroup); // Set the empty bind group at index 0
  renderpass.draw(3, 1, 0, 0); // 3개의 정점으로 삼각형 그리기

  // 렌더패스 종료
  renderpass.end();

  const queue = device.queue;
  queue.submit([commandEncoder.finish()]);
};
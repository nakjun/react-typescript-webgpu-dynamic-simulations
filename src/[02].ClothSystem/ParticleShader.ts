export class ParticleShader {
    particle_shader = `
        struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

        struct VertexInput {
            @location(0) position: vec3<f32>
        };

        struct FragmentOutput {
            @builtin(position) Position: vec4<f32>,
            @location(0) Color: vec4<f32>,
        };

        @vertex
        fn vs_main(vertexInput: VertexInput) -> FragmentOutput {
            var output: FragmentOutput;
            let modelViewProj = transformUBO.projection * transformUBO.view * transformUBO.model;
            output.Position = modelViewProj * vec4<f32>(vertexInput.position, 1.0);
            output.Color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(0.0, 20.0, -1.0));
            let ambient = 1.0;
            let lighting = ambient + (1.0 - ambient); 
            let color = vec3<f32>(in.Color.xyz);
            return vec4<f32>(color * lighting, 1.0);
        }
    `;

    springShader = `
        struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

        struct VertexInput {
            @location(0) position: vec3<f32>
        };

        struct FragmentOutput {
            @builtin(position) Position: vec4<f32>,
            @location(0) Color: vec4<f32>,
        };

        @vertex
        fn vs_main(vertexInput: VertexInput) -> FragmentOutput {
            var output: FragmentOutput;
            let modelViewProj = transformUBO.projection * transformUBO.view * transformUBO.model;
            output.Position = modelViewProj * vec4<f32>(vertexInput.position, 1.0);
            output.Color = vec4<f32>(0.0, 1.0, 0.0, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(0.0, 20.0, 5.0));
            let ambient = 0.1;
            let lighting = ambient + (1.0 - ambient); 
            let color = vec3<f32>(in.Color.xyz);
            return vec4<f32>(color * lighting, 1.0);        
        }
    `;

    textureShader = `
    struct TransformData {
        model: mat4x4<f32>,
        view: mat4x4<f32>,
        projection: mat4x4<f32>,
    };
    @binding(0) @group(0) var<uniform> transformUBO: TransformData;
    @binding(1) @group(0) var myTexture: texture_2d<f32>;
    @binding(2) @group(0) var mySampler: sampler;
    @binding(3) @group(0) var<uniform> cameraPos: vec3<f32>;    
    
    struct LightData {
        position: vec3<f32>,
        color: vec4<f32>,
        intensity: f32,
        specularStrength: f32,
        shininess: f32,
    };
    @binding(4) @group(0) var<uniform> lightUBO: LightData; // Light 정보 바인딩
    @binding(5) @group(0) var<uniform> alpha: f32;
    
    struct VertexOutput {
        @builtin(position) Position : vec4<f32>,
        @location(0) TexCoord : vec2<f32>,
        @location(1) Normal : vec3<f32>,
        @location(2) FragPos : vec3<f32>, // 프래그먼트 위치 추가
    };
    
    @vertex
    fn vs_main(@location(0) vertexPosition: vec3<f32>, @location(1) vertexTexCoord: vec2<f32>, @location(2) vertexNormal: vec3<f32>) -> VertexOutput {
        var output : VertexOutput;
        output.Position = transformUBO.projection * transformUBO.view * transformUBO.model * vec4<f32>(vertexPosition, 1.0);
        output.TexCoord = vertexTexCoord;
        output.Normal = (transformUBO.model * vec4<f32>(vertexNormal, 0.0)).xyz;
        output.FragPos = (transformUBO.model * vec4<f32>(vertexPosition, 1.0)).xyz; // 월드 공간 위치 계산
        return output;
    }
    
    @fragment 
    fn fs_main(@location(0) TexCoord : vec2<f32>, @location(1) Normal : vec3<f32>, @location(2) FragPos: vec3<f32>, @builtin(front_facing) isFrontFacing : bool) -> @location(0) vec4<f32> {
        let norm: vec3<f32> = Normal;        
        let viewDir: vec3<f32> = normalize(cameraPos - FragPos);
        var totalDiffuse: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var totalSpecular: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // 방향성 광원들의 방향
        var directions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
            vec3<f32>(-0.0, 0.0, -1.0),
            vec3<f32>(0.0, 0.0, 1.0),
            vec3<f32>(-0.75, 0.0, 1.0),
            vec3<f32>(0.75, 0.0, -1.0)
        );

        // 각 방향성 광원에 대한 조명 계산 수행
        for (var i = 0; i < 4; i++) {
            let lightDir: vec3<f32> = normalize(-directions[i]);
            let diff: f32 = max(dot(norm, lightDir), 0.0);
            let diffuse: vec4<f32> = lightUBO.color * lightUBO.intensity * diff;
            let spec: f32 = pow(max(dot(viewDir, reflect(-lightDir, norm)), 0.0), lightUBO.shininess);
            let specular: vec4<f32> = lightUBO.color * spec * lightUBO.specularStrength;

            totalDiffuse += diffuse;
            totalSpecular += specular;
        }

        // 텍스처 색상 샘플링
        let textureColor: vec4<f32> = textureSample(myTexture, mySampler, TexCoord);

        // 최종 색상 계산 (텍스처 색상과 조명 효과의 조합)
        var finalColor: vec4<f32> = (totalDiffuse + totalSpecular) * textureColor;

        // if (!isFrontFacing) {
        //     finalColor = vec4<f32>(1.0, 1.0, 1.0, 1.0); // 흰색으로 설정
        // }

        return vec4<f32>(finalColor.rgb, alpha); // Alpha는 지정된 alpha 값으로 설정
    }

    `;

    freefallComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;    
    @group(0) @binding(2) var<storage, read_write> fixed: array<u32>;
    @group(0) @binding(3) var<storage, read_write> force: array<f32>;   
    @group(0) @binding(4) var<storage, read_write> prevPosition: array<f32>;
    @group(0) @binding(5) var<uniform> externalForce : vec3<f32>;    

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getVelocity(index:u32) -> vec3<f32>{
        return vec3<f32>(velocities[index*3],velocities[index*3+1],velocities[index*3+2]);
    }

    fn getForce(index:u32) -> vec3<f32>{
        return vec3<f32>(force[index*3] / 2.0,force[index*3+1] / 2.0,force[index*3+2] / 2.0);
    }

    @compute 
    @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        var fixed = fixed[index];
        
        if(fixed==1){
            return;
        }
        
        var pos = getPosition(index);
        var vel = getVelocity(index);
        var f = getForce(index) * 0.05;     
        
        prevPosition[index*3 + 0] = pos.x;
        prevPosition[index*3 + 1] = pos.y;
        prevPosition[index*3 + 2] = pos.z;
        
        var gravity: vec3<f32> = vec3<f32>(0.0, -4.9, 0.0);        
        var deltaTime: f32 = 0.003; // Assuming 60 FPS for simplicity
        vel += ((f + vec3<f32>(0.0, 0.0, 3.0) + gravity) * deltaTime);
        pos += (vel * deltaTime);
        
        velocities[index*3 + 0] = vel.x;
        velocities[index*3 + 1] = vel.y;
        velocities[index*3 + 2] = vel.z;

        positions[index*3 + 0] = pos.x;
        positions[index*3 + 1] = pos.y;
        positions[index*3 + 2] = pos.z;
    }    
    `;

    getParticleShader() {
        return this.particle_shader;
    }

    getComputeShader() {
        return this.freefallComputeShader;
    }

    getSpringShader() {
        return this.springShader;
    }

    getTextureShader() {
        return this.textureShader;
    }
}

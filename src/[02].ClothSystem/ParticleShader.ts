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
            let ambient = 0.1;
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
    
    // @fragment
    // fn fs_main(@location(0) TexCoord : vec2<f32>, @location(1) Normal : vec3<f32>, @location(2) FragPos: vec3<f32>) -> @location(0) vec4<f32> {

    //     let lightColor = vec4<f32>(0.95, 0.95, 0.9, 1.0);

    //     var shininess = 15.0;

    //     let lightDir = normalize(vec3<f32>(0.0, 0.0, 1.0)); // 빛의 방향
    //     let viewDir = normalize(cameraPos - FragPos); // 뷰(카메라) 방향
    //     let reflectDir = reflect(-lightDir, Normal); // 반사된 빛의 방향
    //     let spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess); // specular 강도 계산
    //     let diff = max(dot(Normal, lightDir), 0.2); // Lambertian 반사율
    //     let ambient = 1.0; // 주변광 설정
    //     let texColor = textureSample(myTexture, mySampler, TexCoord);
    //     let color = texColor * (diff + spec + ambient); // 텍스처 색상에 라이팅 적용
    //     let finalColor = texColor * lightColor * (diff + spec + ambient);
    //     return vec4(finalColor.rgb, 1.0);
    // }

    @fragment
    fn fs_main(@location(0) TexCoord : vec2<f32>, @location(1) Normal : vec3<f32>, @location(2) FragPos: vec3<f32>) -> @location(0) vec4<f32> {    
        var ambientColor: vec4<f32> = vec4<f32>(0.1, 0.1, 0.1, 1.0);
        var finalColor: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        // 조명 설정 예시
        var lights: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
            vec3<f32>(-10.0, 50.0, 10.0), // 조명 1의 위치
            vec3<f32>(10.0, 50.0, 10.0), // 조명 2의 위치
            vec3<f32>(-10.0, 50.0, -10.0), // 조명 3의 위치
            vec3<f32>(10.0, 50.0, -10.0) // 조명 4의 위치
        );

        let shininess: f32 = 128.0;
        let lightColor: vec4<f32> = vec4<f32>(0.95, 0.95, 0.9, 1.0);

        // 텍스처 샘플링
        let texColor: vec4<f32> = textureSample(myTexture, mySampler, TexCoord);

        var shadowFactor: f32 = 1.0;

        // 모든 조명에 대해 반복
        for (var i: i32 = 0; i < 2; i = i + 1) {
            let lightPos: vec3<f32> = lights[i];
            let norm: vec3<f32> = normalize(Normal);
            let lightDir: vec3<f32> = normalize(lightPos - FragPos);
            let diff: f32 = max(dot(norm, lightDir), 0.0);

            
            let specularStrength: f32 = 0.5;
            
            let viewDir: vec3<f32> = normalize(cameraPos - FragPos);
            let reflectDir: vec3<f32> = reflect(-lightDir, norm);
            let spec: f32 = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            
            var shadowValue = dot(viewDir, lightDir);
            let ambient: vec4<f32> = ambientColor * texColor;
            let diffuse: vec4<f32> = (lightColor * diff * texColor) * shadowFactor;
            let specular: vec4<f32> = lightColor * spec * specularStrength * shadowFactor;

            finalColor += ambient + diffuse + specular;
        }

        return finalColor;

    }

    

    `;

    freefallComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;    
    @group(0) @binding(2) var<storage, read_write> fixed: array<u32>;
    @group(0) @binding(3) var<storage, read_write> force: array<f32>;   

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
        var f = getForce(index);        
        
        // floor collisions
        if(pos.y < 0.0){
            pos.y += 0.0001;  
            vel *= -0.001;      
        }

        // // Sphere properties
        // var sphereCenter: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0); // Example sphere center
        // var sphereRadius: f32 = 10.0; // Example sphere radius

        // // Collision detection and response
        // var distanceToSphereCenter = distance(pos, sphereCenter);
        // if(distanceToSphereCenter < sphereRadius){
        //     // Move the particle to the surface of the sphere
        //     var directionToCenter = normalize(sphereCenter - pos);
        //     pos += (-directionToCenter * 0.01);

        //     // Reflect velocity
        //     vel *= -0.001;      
        // }
        
        var gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);
        var deltaTime: f32 = 0.0007; // Assuming 60 FPS for simplicity

        vel += ((f + gravity) * deltaTime);
        pos += (vel * deltaTime);

        velocities[index*3 + 0] = vel.x;
        velocities[index*3 + 1] = vel.y;
        velocities[index*3 + 2] = vel.z;

        positions[index*3 + 0] = pos.x;
        positions[index*3 + 1] = pos.y;
        positions[index*3 + 2] = pos.z;
    }    
    `;

    normalUpdateShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;    
    @group(0) @binding(1) var<storage, read_write> normals: array<f32>;

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getNormals(index:u32) -> vec3<f32>{
        return vec3<f32>(normals[index*3],normals[index*3+1],normals[index*3+2]);
    }

    @compute 
    @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;

        var pos = getPosition(index);
        var normal = getNormals(index);
    }    
    `;


    lineShader = ``;

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

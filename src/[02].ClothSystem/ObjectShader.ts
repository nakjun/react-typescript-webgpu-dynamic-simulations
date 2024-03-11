export class ObjectShader {
    moveComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }

    @compute 
    @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;        
        var pos = getPosition(index);        
        
        pos.x -= 0.007;

        positions[index*3 + 0] = pos.x;
        positions[index*3 + 1] = pos.y;
        positions[index*3 + 2] = pos.z;
    }    
    `;

    getMoveShader() {
        return this.moveComputeShader;
    }

    materialShader = `
    struct TransformData {
        model: mat4x4<f32>,
        view: mat4x4<f32>,
        projection: mat4x4<f32>,
    };
    @binding(0) @group(0) var<uniform> transformUBO: TransformData;    
    @binding(1) @group(0) var<uniform> cameraPos: vec3<f32>;
    @binding(2) @group(0) var<uniform> lightUBO: LightData; // Light 정보 바인딩
    
    struct LightData {
        position: vec3<f32>,
        color: vec4<f32>,
        intensity: f32,
        specularStrength: f32,
        shininess: f32,
    };
    
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
    fn fs_main(@location(0) TexCoord : vec2<f32>, @location(1) Normal : vec3<f32>, @location(2) FragPos: vec3<f32>) -> @location(0) vec4<f32> {            
        
        let lightPos: vec3<f32> = lightUBO.position;
        let lightColor: vec4<f32> = lightUBO.color;
        let lightIntensity: f32 = lightUBO.intensity;
        
        // 주변광 계산
        let ambientColor: vec4<f32> = vec4<f32>(0.319551, 0.435879, 0.802236, 1.0);
    
        // // diffuse 계산
        let norm: vec3<f32> = normalize(Normal);
        let lightDir: vec3<f32> = normalize(lightPos - FragPos);
        let diff: f32 = max(dot(norm, lightDir), 0.0);
        let diffuse: vec4<f32> = lightColor * diff * lightIntensity * vec4<f32>(0.319551, 0.435879, 0.802236, 1.0);
    
        // // specular 계산
        let viewDir: vec3<f32> = normalize(cameraPos - FragPos);
        let reflectDir: vec3<f32> = reflect(-lightDir, norm);
        let spec: f32 = pow(max(dot(viewDir, reflectDir), 0.0), 16);
        let specular: vec4<f32> = lightColor * spec * vec4<f32>(0.429134, 0.429134, 0.429134, 1.0);

        // // 최종 색상 계산
        var finalColor: vec4<f32> = ambientColor + diffuse + specular + vec4<f32>(0.000000, 0.000000, 0.000000, 1.0);        

        return vec4<f32>(finalColor.x,finalColor.y, finalColor.z, 1.0);
        
        //return vec4<f32>(diffuse.x, diffuse.y, diffuse.z, 1.0);
    }
    `;

    getMaterialShader() {
        return this.materialShader;
    }
}
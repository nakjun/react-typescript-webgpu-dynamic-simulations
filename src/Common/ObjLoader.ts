export class ObjModel {
    vertices: number[] = [];
    indices: number[] = [];
    normals: number[] = [];
    uvs: number[] = []; // UVs 배열 추가
}

export class ObjLoader {
    model: ObjModel = new ObjModel();
    parse(objData: string, scale: number = 1.0): ObjModel {
        var _model = new ObjModel();
        const vertexPositions: number[][] = [];
        const vertexNormals: number[][] = [];
        const uv: number[][] = [];
        const faces: string[][] = [];

        objData.split('\n').forEach(line => {
            const parts = line.trim().split(/\s+/);
            switch (parts[0]) {
                case 'v':
                    const scaledPosition = parts.slice(1).map(parseFloat).map(coord => coord * scale);
                    vertexPositions.push(scaledPosition);
                    break;
                case 'vn':
                    vertexNormals.push(parts.slice(1).map(parseFloat));
                    break;
                case 'vt':
                    uv.push(parts.slice(1).map(parseFloat));
                    break;
                case 'f':
                    faces.push(parts.slice(1));
                    break;
            }
        });

        const indexMap = new Map<string, number>();
        let currentIndex = 0;

        faces.forEach(faceParts => {
            const faceIndices = faceParts.map(part => {
                const [pos, tex, norm] = part.split('/').map(e => parseInt(e) - 1);
                const key = `${pos}|${tex}|${norm}`;

                if (indexMap.has(key)) {
                    return indexMap.get(key)!;
                } else {
                    const position = vertexPositions[pos];
                    _model.vertices.push(...position);

                    // UVs를 (0, 0)으로 설정
                    const _uv = uv[tex];
                    if(_uv===undefined){
                        _model.uvs.push(0.01, 0.01);
                    }
                    else{
                        _model.uvs.push(..._uv);
                    }

                    if (vertexNormals.length > 0 && norm !== undefined) {
                        const normal = vertexNormals[norm];
                        _model.normals.push(...normal);
                    }

                    indexMap.set(key, currentIndex);
                    return currentIndex++;
                }
            });

            for (let i = 1; i < faceIndices.length - 1; i++) {
                _model.indices.push(faceIndices[0], faceIndices[i], faceIndices[i + 1]);
            }
        });

        // 정점 법선이 없는 경우 계산
        if (vertexNormals.length === 0) {
            this.calculateNormals(_model.vertices, _model.indices).forEach(n => _model.normals.push(n));
        }

        console.log("parse end");

        return _model;
    }

    calculateNormals(vertices: number[], indices: number[]): number[] {
        const normals = new Array(vertices.length / 3).fill(0).map(() => [0, 0, 0]);

        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i];
            const i1 = indices[i + 1];
            const i2 = indices[i + 2];

            const v1 = vertices.slice(i0 * 3, i0 * 3 + 3);
            const v2 = vertices.slice(i1 * 3, i1 * 3 + 3);
            const v3 = vertices.slice(i2 * 3, i2 * 3 + 3);

            const u = v2.map((val, idx) => val - v1[idx]);
            const v = v3.map((val, idx) => val - v1[idx]);

            const norm = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]
            ];

            // 정점별 법선 누적
            for (let j = 0; j < 3; j++) {
                normals[i0][j] += norm[j];
                normals[i1][j] += norm[j];
                normals[i2][j] += norm[j];
            }
        }

        // 누적된 법선 정규화
        const normalizedNormals = normals.flatMap(norm => {
            const len = Math.sqrt(norm[0] ** 2 + norm[1] ** 2 + norm[2] ** 2);
            return [norm[0] / len, norm[1] / len, norm[2] / len];
        });

        return normalizedNormals;
    }

    async load(url: string, scale: number = 1.0): Promise<ObjModel> {
        const response = await fetch(url);
        const objData = await response.text();
        return this.parse(objData, scale);
    }
}

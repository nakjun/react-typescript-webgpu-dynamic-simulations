import { vec3 } from "gl-matrix";

export class Triangle{
    
    v1!:number;
    v2!:number;
    v3!:number;

    constructor(v1: number, v2: number, v3: number){
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
    }
}

export class Node {
    position!: vec3;
    velocity!: vec3;
    acceleration!: vec3;

    fixed: boolean = false;

    springs: Spring[] = [];
    triangles: Triangle[] = [];

    constructor(pos: vec3, vel: vec3) {
        this.position = pos;
        this.velocity = vel;
        this.acceleration = vec3.create();
        this.fixed = false;
    }
}

export class Spring {
    n1!: Node;
    n2!: Node;
    mRestLen: number = 0;

    kS: number = 1000.0;
    kD: number = 0.01;
    type: string = "spring type";

    index1: number = 0;
    index2: number = 0;

    targetIndex1: number = 0;
    targetIndex2: number = 0;

    constructor(_n1: Node, _n2: Node, ks: number, kd: number, type: string, _i1: number, _i2: number) {
        this.n1 = _n1;
        this.n2 = _n2;

        this.kS = ks
        this.kD = kd;
        this.type = type;

        this.mRestLen = Math.round((vec3.distance(this.n1.position, this.n2.position) + Number.EPSILON) * 100) / 100;
        this.index1 = _i1;
        this.index2 = _i2;
    }
}
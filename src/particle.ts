export class Particle {    
    position: [number, number, number];
    velocity: [number, number, number];

    constructor(position: [number, number, number], velocity: [number, number, number] = [0, 0, 0]) {
        this.position = position;
        this.velocity = velocity;
    }
}
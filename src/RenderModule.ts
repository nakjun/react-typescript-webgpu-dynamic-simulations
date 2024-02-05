import { Renderer } from "./[01].ParticleSystem/Renderer";
import { ClothRenderer } from "./[02].ClothSystem/Renderer";


export const Initialize = async () => {    
  const sceneManager = new ClothRenderer("gfx-main");
  sceneManager.init().then(() => {    
    sceneManager.createClothModel(64, 64, 1000, 0.01);
    sceneManager.createClothBuffers();
    sceneManager.createParticlePipeline();
    sceneManager.createSpringPipeline();

    animate();
  });

  // Create an animation loop function
  function animate() {
    // Update the particle simulation (e.g., call your compute shader)
    //sceneManager.dispatchComputeShader();

    // Render the scene (e.g., call your render function)
    sceneManager.render();

    // Request the next frame
    requestAnimationFrame(animate);
  }
}
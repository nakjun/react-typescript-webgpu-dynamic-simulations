import { Renderer } from "./[01].ParticleSystem/Renderer";
import { ClothRenderer } from "./[02].ClothSystem/Renderer";


export const Initialize = async () => {    
  // const sceneManager = new Renderer("gfx-main");
  // sceneManager.init().then(() => {   
  //   sceneManager.createBuffers();
  //   sceneManager.createPipeline();
  //   sceneManager.createParticles(100000);
  //   sceneManager.createParticleBuffers();
  //   sceneManager.createParticlePipeline();
  //   sceneManager.createComputePipeline();
  //   sceneManager.createComputeBindGroup();
  //   animate();
  // });


  const sceneManager = new ClothRenderer("gfx-main");
  sceneManager.init().then(() => {    
    sceneManager.createClothModel(128, 128, 700.0, 0.1);    
    sceneManager.createClothBuffers();
    sceneManager.createRenderPipeline();
    sceneManager.createSpringPipeline();
    sceneManager.createParticlePipeline();
    sceneManager.createSpringForceComputePipeline();
    sceneManager.createNodeForceSummationPipeline();
    
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
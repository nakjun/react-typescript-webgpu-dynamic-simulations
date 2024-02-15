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

  const canvas = document.querySelector("canvas#gfx-main") as HTMLCanvasElement; // `as HTMLCanvasElement`로 타입 단언 사용
  if (!canvas) {
    console.error("Canvas element not found");
    return;
  }

  let isLeftMouseDown = false;
  let isRightMouseDown = false;
  let lastMouseX: number, lastMouseY: number;

  canvas.addEventListener('mousedown', (event: MouseEvent) => { // `MouseEvent` 타입 명시
    if (event.button === 0) { // 좌클릭
      isLeftMouseDown = true;
      console.log("좌클릭");
    } else if (event.button === 2) { // 우클릭
      isRightMouseDown = true;
      console.log("우클릭");
    }
    lastMouseX = event.clientX;
    lastMouseY = event.clientY;
    
  });

  document.addEventListener('mouseup', (event) => {
    isLeftMouseDown = false;
    isRightMouseDown = false;
});



  const sceneManager = new ClothRenderer("gfx-main");
  sceneManager.init().then(() => {

    canvas.addEventListener('mousemove', (event) => {
      if (isLeftMouseDown) {
          // 카메라 회전 로직 구현
          const dx = event.clientX - lastMouseX;
          const dy = event.clientY - lastMouseY;
          //console.log("rotate");
          sceneManager.rotateCamera(dx, dy);
  
      } else if (isRightMouseDown) {
          // 카메라 패닝 로직 구현
          const dx = event.clientX - lastMouseX;
          const dy = event.clientY - lastMouseY;
          console.log("pan");
      }
      lastMouseX = event.clientX;
      lastMouseY = event.clientY;
  });
  
  canvas.addEventListener('wheel', (event) => {
      // 카메라 줌 인/아웃 로직 구현
      sceneManager.zoomCamera(event.deltaY / 100);      
  });
    //sceneManager.createClothModel(4, 4, 200.0, 150.0, 1000.0, 0.5);  
    //sceneManager.createClothModel(16, 16, 500.0, 250.0, 1500.0, 0.3);
    //sceneManager.createClothModel(256, 256, 5000.0, 1550.0, 100000.0, 0.1);
    //sceneManager.createClothModel(400, 400, 4000.0, 3500.0, 5500.0, 0.1);
    sceneManager.createSphereModel();  
    sceneManager.createClothModel(512, 512, 10000.0, 5600.0, 75000.0, 0.001);
    //sceneManager.createClothModel(750, 750, 7000.0, 5000.0, 20000.0, 0.03);
    //sceneManager.createClothModel(850, 850, 15000.0, 9500.0, 70000.0, 0.001);
    sceneManager.createClothBuffers();    
    sceneManager.createRenderPipeline();
    sceneManager.createSpringPipeline();
    sceneManager.createTrianglePipeline();
    sceneManager.createParticlePipeline();
    sceneManager.createUpdateNormalPipeline();
    sceneManager.createSpringForceComputePipeline();
    sceneManager.createNodeForceSummationPipeline();

    animate();
  });

  // Create an animation loop function
  function animate() {    
    sceneManager.render();
    requestAnimationFrame(animate);
  }
}
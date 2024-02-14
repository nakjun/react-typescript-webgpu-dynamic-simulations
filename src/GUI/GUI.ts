import GUI from 'lil-gui';

export class SystemGUI{
    performanceGui!: GUI;
    renderOptionGui!: GUI;

    constructor(){
        this.performanceGui = new GUI({
            container: document.body,
            autoPlace: false // 기본 위치 배치 사용 안함
        });
        this.performanceGui.title("Performance");
        this.performanceGui.domElement.style.position = 'absolute';
        this.performanceGui.domElement.style.top = '0px';
        this.performanceGui.domElement.style.left = '0px';        

        this.renderOptionGui = new GUI({
            container: document.body,
            autoPlace: false // 기본 위치 배치 사용 안함
        });
        this.renderOptionGui.title("Render Options");
        this.renderOptionGui.domElement.style.position = 'absolute';
        this.renderOptionGui.domElement.style.top = '100px';
        this.renderOptionGui.domElement.style.left = '0px';      
    }
}
package globalquake.ui.globe.feature;

public class RenderEntity<E> {

    private final E original;

    private final RenderElement[] renderElements;

    public RenderEntity(E original, int renderElements) {
        this.original = original;
        this.renderElements = new RenderElement[renderElements];
        for (int i = 0; i < renderElements; i++) {
            this.renderElements[i] = new RenderElement();
        }
    }

    public E getOriginal() {
        return original;
    }

    public RenderElement getRenderElement(int index) {
        return renderElements[index];
    }

    public RenderElement[] getRenderElements() {
        return renderElements;
    }
}

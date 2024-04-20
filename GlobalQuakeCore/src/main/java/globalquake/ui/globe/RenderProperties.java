package globalquake.ui.globe;

public class RenderProperties {
    public final double centerLat;
    public final double centerLon;
    public final double scroll;

    public final int width;

    public final int height;
    private RenderPrecomputedValues renderPrecomputedValues;

    public RenderProperties(int width, int height, double centerLat, double centerLon, double scroll) {
        this.width = width;
        this.height = height;
        this.centerLat = centerLat;
        this.centerLon = centerLon;
        this.scroll = scroll;
    }

    public void setPrecomputed(RenderPrecomputedValues renderPrecomputedValues) {
        this.renderPrecomputedValues = renderPrecomputedValues;
    }

    public RenderPrecomputedValues getRenderPrecomputedValues() {
        return renderPrecomputedValues;
    }
}

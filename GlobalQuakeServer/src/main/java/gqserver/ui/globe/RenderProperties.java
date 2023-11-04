package gqserver.ui.globe;

@SuppressWarnings("ClassCanBeRecord")
public class RenderProperties{
    public final double centerLat;
    public final double centerLon;
    public final double scroll;

    public final int width;

    public final int height;

    public RenderProperties(int width, int height, double centerLat, double centerLon, double scroll) {
        this.width = width;
        this.height = height;
        this.centerLat = centerLat;
        this.centerLon = centerLon;
        this.scroll = scroll;
    }

}

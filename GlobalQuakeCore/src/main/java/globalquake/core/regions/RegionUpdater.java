package globalquake.core.regions;

public class RegionUpdater {

    public static final String DEFAULT_REGION = "Pending...";

    private final Regional target;

    public RegionUpdater(Regional target) {
        this.target = target;
        if (target.getRegion() == null) {
            target.setRegion(DEFAULT_REGION);
        }
    }

    public void updateRegion() {
        target.setRegion(Regions.getRegion(target.getLat(), target.getLon()));
    }
}

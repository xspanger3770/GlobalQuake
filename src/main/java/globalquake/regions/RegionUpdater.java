package globalquake.regions;

import java.util.Objects;

public class RegionUpdater {

    public static final String DEFAULT_REGION = "Pending...";

    private final Regional target;
    private boolean regionUpdateRunning = false;

    public RegionUpdater(Regional target) {
        this.target = target;
        if(target.getRegion() == null) {
            target.setRegion(DEFAULT_REGION);
        }
    }

    public void updateRegion() {
        if (regionUpdateRunning) {
            return;
        }

        target.setRegion(Regions.getRegion(target.getLat(), target.getLon()));

        new Thread("Region Search") {
            public void run() {
                regionUpdateRunning = true;
                try {
                    String newRegion = Regions.downloadRegion(target.getLat(), target.getLon());
                    if (!Objects.equals(newRegion, Regions.UNKNOWN_REGION)) {
                        target.setRegion(newRegion);
                    }
                }finally {
                    regionUpdateRunning = false;
                }
            }
        }.start();
    }
}

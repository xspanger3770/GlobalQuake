package globalquake.playground;

import globalquake.client.GlobalQuakeLocal;
import globalquake.core.GlobalQuake;
import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.regions.Regions;
import globalquake.main.Main;
import globalquake.utils.Scale;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import org.tinylog.Logger;

import java.awt.*;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Collection;

public class GlobalQuakePlayground extends GlobalQuakeLocal {

    public long createdAtMillis;
    private final long playgroundStartMillis = LocalDate.of(2000,1,1)
            .atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();

    private final WaveformGenerator waveformGenerator;

    private final Collection<Earthquake> playgroundEarthquakes = new MonitorableCopyOnWriteArrayList<>();

    public static void main(String[] args) throws Exception{
        GlobalQuake.prepare(Main.MAIN_FOLDER, new ApplicationErrorHandler(null));
        Regions.init();
        Scale.load();

        new GlobalQuakePlayground();
    }

    @Override
    public void startRuntime() {
        getGlobalQuakeRuntime().runThreads();
    }

    public GlobalQuakePlayground() {
        super(new StationDatabaseManagerPlayground(), new GlobalStationManagerPlayground());
        this.waveformGenerator = new WaveformGenerator(this);
        createdAtMillis = System.currentTimeMillis();
        createFrame();
        startRuntime();
    }

    public GlobalQuakePlayground createFrame() {
        EventQueue.invokeLater(() -> {
            try {
                globalQuakeFrame = new GlobalQuakeFramePlayground();
                globalQuakeFrame.setVisible(true);

                Main.getErrorHandler().setParent(globalQuakeFrame);
            }catch (Exception e){
                Logger.error(e);
                System.exit(0);
            }
        });
        return this;
    }

    @Override
    public long currentTimeMillis() {
        return playgroundStartMillis + (System.currentTimeMillis() - createdAtMillis);
    }

    @Override
    public EarthquakeArchive createArchive() {
        return new EarthquakeArchive();
    }

    public Collection<Earthquake> getPlaygroundEarthquakes() {
        return playgroundEarthquakes;
    }
}

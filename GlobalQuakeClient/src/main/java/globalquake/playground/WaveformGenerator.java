package globalquake.playground;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.GlobalQuake;
import globalquake.core.station.AbstractStation;
import org.tritonus.share.GlobalInfo;

import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class WaveformGenerator {
    private final GlobalQuakePlayground globalQuakePlayground;

    public WaveformGenerator(GlobalQuakePlayground globalQuakePlayground) {
        this.globalQuakePlayground = globalQuakePlayground;
        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(this::updateWaveforms, 0, 250, TimeUnit.MILLISECONDS);
    }

    private void updateWaveforms() {
        for(AbstractStation station : globalQuakePlayground.getStationManager().getStations()) {
            PlaygroundStation playgroundStation = (PlaygroundStation) station;
            long lastLog = playgroundStation.lastSampleTime;
            long now = GlobalQuake.instance.currentTimeMillis();

            if(lastLog < 0){
                lastLog = now;
            }


            while(lastLog < now) {
                station.getAnalysis().nextSample(
                        playgroundStation.getNoise(lastLog),
                        lastLog,
                        GlobalQuake.instance.currentTimeMillis());
                lastLog += 1000.0 / station.getAnalysis().getSampleRate();
            }
            playgroundStation.lastSampleTime = lastLog;
        }
    }
}

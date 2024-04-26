package globalquake.playground;

import globalquake.core.GlobalQuake;
import globalquake.core.station.AbstractStation;
import org.tinylog.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class WaveformGenerator {
    private final GlobalQuakePlayground globalQuakePlayground;

    public WaveformGenerator(GlobalQuakePlayground globalQuakePlayground) {
        this.globalQuakePlayground = globalQuakePlayground;
        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(this::updateWaveforms, 0, 250, TimeUnit.MILLISECONDS);
    }

    private void updateWaveforms() {
        try {
            globalQuakePlayground.getStationManager().getStations().parallelStream().forEach(WaveformGenerator::generateWaveform);
        } catch (Exception e) {
            Logger.error(e);
        }
    }

    private static void generateWaveform(AbstractStation station) {
        PlaygroundStation playgroundStation = (PlaygroundStation) station;
        long lastLog = playgroundStation.lastSampleTime;
        long now = GlobalQuake.instance.currentTimeMillis() - playgroundStation.getDelay();

        if (lastLog < 0) {
            lastLog = now - 2 * 60 * 1000;
        }


        while (lastLog < now) {
            station.getAnalysis().nextSample(
                    playgroundStation.getNoise(lastLog),
                    lastLog,
                    GlobalQuake.instance.currentTimeMillis());
            lastLog += (long) (1000.0 / station.getAnalysis().getSampleRate());
        }
        playgroundStation.lastSampleTime = lastLog;
    }
}

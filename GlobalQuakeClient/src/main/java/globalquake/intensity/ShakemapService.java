package globalquake.intensity;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvValidationException;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeArchiveEvent;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.intensity.CityIntensity;
import globalquake.core.intensity.IntensityScales;
import globalquake.events.specific.ShakeMapsUpdatedEvent;
import globalquake.client.GlobalQuakeLocal;
import globalquake.core.intensity.CityLocation;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ShakemapService {

    private final Map<UUID, ShakeMap> shakeMaps = new HashMap<>();

    private final ExecutorService shakemapService = Executors.newSingleThreadExecutor();
    private final ScheduledExecutorService checkService = Executors.newSingleThreadScheduledExecutor();

    private static final List<CityLocation> cities = new ArrayList<>();

    static {
        load();
    }

    private static void load() {
        int errors = 0;
        try (CSVReader reader = new CSVReaderBuilder(new InputStreamReader(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("cities/worldcities.csv")).openStream())).withSkipLines(1).build()) {
            String[] fields;
            while ((fields = reader.readNext()) != null) {
                String cityName = fields[1];
                double lat = Double.parseDouble(fields[2]);
                double lon = Double.parseDouble(fields[3]);

                int population;

                try {
                    population = Integer.parseInt(fields[9]);
                } catch (Exception e) {
                    population = -1;
                    errors++;
                }

                cities.add(new CityLocation(cityName, lat, lon, population));
            }
        } catch (IOException | CsvValidationException e) {
            Logger.error(e);
        }

        Logger.warn("%d cities have unknown population!".formatted(errors));
    }

    public ShakemapService() {
        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                updateShakemap(event.earthquake());
            }

            @Override
            public void onQuakeArchive(QuakeArchiveEvent event) {
                removeShakemap(event.archivedQuake().getUuid());
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                updateShakemap(event.earthquake());
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                removeShakemap(event.earthquake().getUuid());
            }
        });

        checkService.scheduleAtFixedRate(this::checkShakemaps, 0, 1, TimeUnit.MINUTES);
    }

    private void checkShakemaps() {
        try {
            for (Iterator<Map.Entry<UUID, ShakeMap>> iterator = shakeMaps.entrySet().iterator(); iterator.hasNext(); ) {
                var kv = iterator.next();
                UUID uuid = kv.getKey();
                if (GlobalQuake.instance.getEarthquakeAnalysis().getEarthquake(uuid) == null) {
                    iterator.remove();
                }
            }
        } catch (Exception e) {
            Logger.error(e);
        }
    }

    private void removeShakemap(UUID uuid) {
        shakemapService.submit(() -> {
            try {
                shakeMaps.remove(uuid);
                GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new ShakeMapsUpdatedEvent());
            } catch (Exception e) {
                Logger.error(e);
            }
        });
    }

    private void updateShakemap(Earthquake earthquake) {
        shakemapService.submit(() -> {
            try {
                shakeMaps.put(earthquake.getUuid(), createShakemap(earthquake));
                GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new ShakeMapsUpdatedEvent());
                updateCities(earthquake);
            } catch (Exception e) {
                Logger.error(e);
            }
        });
    }

    private void updateCities(Earthquake earthquake) {
        List<CityIntensity> result = new ArrayList<>();
        double threshold = IntensityScales.getIntensityScale().getLevels().get(0).getPga();

        cities.forEach(cityLocation -> {
            double pga = calculatePGA(cityLocation, earthquake);
            if (pga >= threshold) {
                result.add(new CityIntensity(cityLocation, pga));
            }
        });

        result.sort(Comparator.comparing(cityIntensity -> -cityIntensity.pga()));

        earthquake.cityIntensities = result;
    }

    private double calculatePGA(CityLocation cityLocation, Earthquake earthquake) {
        double dist = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(), -earthquake.getDepth(),
                cityLocation.lat(), cityLocation.lon(), 0);
        return GeoUtils.pgaFunction(earthquake.getMag(), dist, earthquake.getDepth());
    }

    private ShakeMap createShakemap(Earthquake earthquake) {
        Hypocenter hyp = earthquake.getCluster().getPreviousHypocenter();
        double mag = hyp.magnitude + hyp.depth / 200.0;
        mag += Settings.shakemapQualityOffset;
        return new ShakeMap(hyp, mag <= 4.9 ? 6 : mag < 6.4 ? 5 : mag < 8.5 ? 4 : 3);
    }

    public void stop() {
        GlobalQuake.instance.stopService(shakemapService);
        GlobalQuake.instance.stopService(checkService);
    }

    public Map<UUID, ShakeMap> getShakeMaps() {
        return shakeMaps;
    }

    public void clear() {
        shakeMaps.clear();
    }
}

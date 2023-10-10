package globalquake.core.earthquake;

import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.earthquake.data.Earthquake;
import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class EarthquakeArchiveTest {

    @Test
    public void testRaceCondition() {
        EarthquakeArchive earthquakeArchive = new EarthquakeArchive();
        Random r = new Random();
        AtomicInteger n = new AtomicInteger(0);
        for (int i = 0; i < 10000; i++) {
            List<Integer> list = List.of(0, 1, 2, 3, 4);
            list.parallelStream().forEach(integer -> {
                Earthquake earthquake = new Earthquake(null, 0, 0, 0, r.nextLong());
                earthquakeArchive.archiveQuake(earthquake);
                n.incrementAndGet();
            });
            list.parallelStream().forEach(integer -> {
                for(ArchivedQuake archivedQuake : earthquakeArchive.getArchivedQuakes()){
                    archivedQuake.setWrong(!archivedQuake.isWrong());
                }
            });
        }

        assertEquals(50000, n.get());
    }

}
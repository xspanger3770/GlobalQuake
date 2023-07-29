package globalquake.simulator;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;
import com.morce.globalquake.database.StationDatabase;

import globalquake.core.AlertManager;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.Earthquake;
import globalquake.core.earthquake.EarthquakeArchive;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.Event;
import globalquake.core.station.GlobalStation;
import globalquake.core.station.NearbyStationDistanceInfo;
import globalquake.ui.EarthquakeListPanel;
import globalquake.geo.GeoUtils;
import globalquake.geo.IntensityTable;
import globalquake.utils.NamedThreadFactory;
import globalquake.geo.TravelTimeTable;

@SuppressWarnings("CallToPrintStackTrace")
public class EarthquakeSimulator extends JFrame {

    public final Object earthquakesSync = new Object();
    private ArrayList<SimulatedEarthquake> earthquakes;
    private ArrayList<SimulatedStation> stations;
    private FakeGlobalQuake fakeGlobalQuake;

    protected long lastEQSim;
    protected long lastGC;
    protected long lastClusters;
    private ClusterAnalysis clusterAnalysis;
    private EarthquakeAnalysis earthquakeAnalysis;
    public long lastQuakesT;
    private AlertManager alertManager;

    public static final double P_INACCURACY = 1000;


    public static final int RAYS = 9;

    /**
     * 0.0005 - low 0.001 - medium 0.002 - high 0.004 - savage in reality this is
     * very low (maybe about 0.001 depending on time)
     */
    public static final double SHIT_PER_SECOND = 0.001;

    public EarthquakeSimulator() {
        init();
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        EarthquakeSimulatorPanel panel = new EarthquakeSimulatorPanel(this);
        EarthquakeListPanel list = new EarthquakeListPanel(getFakeGlobalQuake());
        panel.setPreferredSize(new Dimension(600, 600));
        list.setPreferredSize(new Dimension(300, 600));

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        mainPanel.setPreferredSize(new Dimension(800, 600));
        mainPanel.add(panel, BorderLayout.CENTER);
        mainPanel.add(list, BorderLayout.EAST);

        setContentPane(mainPanel);

        pack();
        setLocationRelativeTo(null);
        setMinimumSize(new Dimension(600, 500));
        setResizable(true);
        setTitle("GlobalQuake Sim");

        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                mainPanel.repaint();
            }
        }, 0, 50);

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    synchronized (earthquakesSync) {
                        earthquakes.clear();
                    }
                    for (SimulatedStation stat : getStations()) {
                        stat.getAnalysis().getDetectedEvents().clear();
                    }
                    getClusterAnalysis().getClusters().clear();
                    earthquakeAnalysis.getEarthquakes().clear();
                } else if (e.getKeyCode() == KeyEvent.VK_BACK_SPACE) {
                    long skip = 3 * 1000;
                    synchronized (earthquakesSync) {
                        for (SimulatedEarthquake sim : earthquakes) {
                            sim.setOrigin(sim.getOrigin() - skip);
                        }
                    }
                    for (SimulatedStation simStat : stations) {
                        for (Event ev : simStat.getAnalysis().getDetectedEvents()) {
                            ev.setpWave(ev.getpWave() - skip);
                        }
                    }


                    for (Earthquake ea : getFakeGlobalQuake().getEarthquakeAnalysis().getEarthquakes()) {
                        ea.setOrigin(ea.getOrigin() - skip);
                    }
                }
            }
        });
    }

    private void init() {
        createStations();
        createListOfClosestStations();
        earthquakes = new ArrayList<>();
        fakeGlobalQuake = new FakeGlobalQuake(this);
        runThreads();
    }

    private void createListOfClosestStations() {
        for (GlobalStation stat : stations) {
            ArrayList<ArrayList<StationDistanceInfo>> rays = new ArrayList<>();
            for (int i = 0; i < RAYS; i++) {
                rays.add(new ArrayList<>());
            }
            int num = 0;
            for (int i = 0; i < 2; i++) {
                for (GlobalStation stat2 : stations) {
                    if (!(stat2.getId() == stat.getId())) {
                        double dist = GeoUtils.greatCircleDistance(stat.getLat(), stat.getLon(), stat2.getLat(),
                                stat2.getLon());
                        if (dist > (i == 0 ? 1200 : 3600)) {
                            continue;
                        }
                        double ang = GeoUtils.calculateAngle(stat.getLat(), stat.getLon(), stat2.getLat(),
                                stat2.getLon());
                        int ray = (int) ((ang / 360.0) * (RAYS - 1.0));
                        rays.get(ray).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        int ray2 = ray + 1;
                        if (ray2 == RAYS) {
                            ray2 = 0;
                        }
                        int ray3 = ray - 1;
                        if (ray3 == -1) {
                            ray3 = RAYS - 1;
                        }
                        rays.get(ray2).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        rays.get(ray3).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        num++;
                    }
                }
                if (num > 4) {
                    break;
                }
            }
            ArrayList<Integer> closestStations = new ArrayList<>();
            ArrayList<NearbyStationDistanceInfo> nearby = new ArrayList<>();
            for (int i = 0; i < RAYS; i++) {
                if (!rays.get(i).isEmpty()) {
                    rays.get(i).sort(Comparator.comparing(StationDistanceInfo::dist));
                    for (int j = 0; j <= Math.min(1, rays.get(i).size() - 1); j++) {
                        if (!closestStations.contains(rays.get(i).get(j).id())) {
                            closestStations.add(rays.get(i).get(j).id());
                            nearby.add(new NearbyStationDistanceInfo(getStations().get(rays.get(i).get(j).id()),
                                    rays.get(i).get(j).dist(), rays.get(i).get(j).ang()));
                        }
                    }
                }
            }
            stat.setNearbyStations(nearby);
        }
    }

    private void runThreads() {
        ScheduledExecutorService execSim = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Simulated Earthquake Simulation Thread"));
        ScheduledExecutorService execClusters = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Simulated Cluster Analysis Thread"));
        ScheduledExecutorService execGC = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Actual Garbage Collector Thread"));
        ScheduledExecutorService execQuake = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Simulated Hypocenter Location Thread"));

        clusterAnalysis = new ClusterAnalysis(getFakeGlobalQuake());
        earthquakeAnalysis = new EarthquakeAnalysis(getFakeGlobalQuake());
        alertManager = new AlertManager(getFakeGlobalQuake());
        getFakeGlobalQuake().archive = new EarthquakeArchive(getFakeGlobalQuake());

        getFakeGlobalQuake().earthquakeAnalysis = earthquakeAnalysis;
        getFakeGlobalQuake().clusterAnalysis = clusterAnalysis;
        getFakeGlobalQuake().alertManager = alertManager;

        execSim.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                updateEarthquakes();
                shit();
                lastEQSim = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occured in earthquake simulations");
                e.printStackTrace();
            }
        }, 0, 500, TimeUnit.MILLISECONDS);

        execGC.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                System.gc();
                lastGC = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception in garbage collector");
                e.printStackTrace();
            }
        }, 0, 1, TimeUnit.MINUTES);
        execClusters.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                clusterAnalysis.run();
                lastClusters = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occured cluster analysis loop");
                e.printStackTrace();
            }
        }, 0, 500, TimeUnit.MILLISECONDS);
        execQuake.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                earthquakeAnalysis.run();
                alertManager.tick();
                lastQuakesT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occured in hypocenter location loop");
                e.printStackTrace();
            }
        }, 0, 1, TimeUnit.SECONDS);
    }

    protected void shit() {
        for (GlobalStation station : getStations()) {
            if (random.nextDouble() < (SHIT_PER_SECOND / 10.0)) {
                Event shitEvent = new Event(station.getAnalysis());
                shitEvent.setpWave(System.currentTimeMillis());
                shitEvent.maxRatio = random.nextDouble() * 32.0;
                station.getAnalysis().getDetectedEvents().add(shitEvent);

                new Thread(() -> {
                    try {
                        Thread.sleep((long) (random.nextDouble() * 30000));
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    shitEvent.end(System.currentTimeMillis());
                }).start();
            }
        }
    }

    public final Random random = new Random();

    @SuppressWarnings("unused")
    protected void updateEarthquakes() {
        synchronized (earthquakesSync) {
            for (SimulatedEarthquake simE : getEarthquakes()) {
                long age = System.currentTimeMillis() - simE.getOrigin();
                double pDist = TravelTimeTable.getPWaveTravelAngle(simE.getDepth(), age / 1000.0, false) / 360.0
                        * GeoUtils.EARTH_CIRCUMFERENCE;
                double sDist = TravelTimeTable.getSWaveTravelAngle(simE.getDepth(), age / 1000.0, false) / 360.0
                        * GeoUtils.EARTH_CIRCUMFERENCE;
                double dDist = (2.0 / Math.pow(simE.getMag(), 2.0)) * (age / 1000.0 - 25 - Math.pow(simE.getMag(), 2));
                for (SimulatedStation simStat : getStations()) {
                    double distGC = GeoUtils.greatCircleDistance(simE.getLat(), simE.getLon(), simStat.getLat(),
                            simStat.getLon());
                    // double distAng = (distGC * 360.0) / GeoUtils.EARTH_CIRCUMFERENCE;
                    double distGEO = GeoUtils.geologicalDistance(simE.getLat(), simE.getLon(), -simE.getDepth(),
                            simStat.getLat(), simStat.getLon(), simStat.getAlt() / 1000.0);
                    double maxR = IntensityTable.getMaxIntensity(simE.getMag(), distGEO) * simStat.getSensFactor();
                    long actualPWave = simE.getOrigin() + (long) (1000
                            * TravelTimeTable.getPWaveTravelTime(simE.getDepth(), TravelTimeTable.toAngle(distGC)));
                    long actualSWave = simE.getOrigin() + (long) (1000
                            * TravelTimeTable.getSWaveTravelTime(simE.getDepth(), TravelTimeTable.toAngle(distGC)));

                    maxR *= Math.min(1, (System.currentTimeMillis() - actualPWave) / (Math.pow(4, simE.getMag())));

                    if (!simE.getArrivedPWave().contains(simStat)) {
                        if (distGC < pDist) {
                            Event event = new Event(simStat.getAnalysis());
                            actualPWave += (long) (((random.nextDouble() - 0.5) * 2) * P_INACCURACY);
                            event.setpWave(actualPWave);
                            event.maxRatio = maxR;
                            simE.getArrivedPWave().add(simStat);
                            boolean b;
                            if (maxR > 10) {
                                b = true;
                            } else if (maxR < 3.0) {
                                b = false;
                            } else {
                                b = random.nextDouble() < (maxR - 3.0) / 7.0;
                            }
                            if (b) {
                                simStat.getAnalysis().getDetectedEvents().add(0, event);
                                simE.getEventMap().put(simStat, event);
                            }
                        }
                    }
                    if (distGC < pDist) {
                        Event event = simE.getEventMap().get(simStat);
                        if (event != null) {
                            event.maxRatio = maxR;
                        }
                    }
                    if (distGC < sDist) {
                        Event event = simE.getEventMap().get(simStat);
                        double _maxR = maxR * (2 - distGC / 400.0);
                        if (event == null) {
                            event = new Event(simStat.getAnalysis());
                            event.setpWave(actualPWave);
                            event.setsWave(actualSWave);
                            simE.getEventMap().put(simStat, event);
                            boolean b;
                            if (_maxR > 10) {
                                b = true;
                            } else if (_maxR < 3.0) {
                                b = false;
                            } else {
                                b = random.nextDouble() < (_maxR - 3.0) / 7.0;
                            }
                            if (b) {
                                simStat.getAnalysis().getDetectedEvents().add(0, event);
                            }
                        } else {
                            event.setsWave(actualSWave);
                        }
                        simE.getArrivedSWave().add(simStat);
                        if (_maxR > event.maxRatio) {
                            event.maxRatio = _maxR;
                        }
                    }
                    if (dDist > distGC) {
                        Event e = simE.getEventMap().get(simStat);
                        if (e != null && !e.hasEnded()) {
                            e.end(System.currentTimeMillis());
                            simStat.getAnalysis().getDetectedEvents().remove(e);
                        }

                    }
                }
            }
        }
    }

    private void createStations() {
        stations = new ArrayList<>();
        try {
            ObjectInputStream in = new ObjectInputStream(
                    new FileInputStream("./GlobalQuake/stationDatabase/stationDatabaseNormal.dat"));
            StationDatabase database = (StationDatabase) in.readObject();
            in.close();

            doImportantStuff(database);

            int i = 0;
            database.getNetworksReadLock().lock();
            try {
                for (Network n : database.getNetworks()) {
                    for (Station s : n.getStations()) {
                        Channel ch = s.getChannels().get(0);
                        if (database.getSelectedStation(s) != null) {
                            stations.add(new SimulatedStation(n.getNetworkCode(), s.getStationCode(), ch.getName(),
                                    ch.getLocationCode(), ch.getSeedlinkNetwork(), ch.getSource(), s.getLat(), s.getLon(),
                                    s.getAlt(), ch.getSensitivity(), ch.getFrequency(), i, 1));
                            i++;
                        }
                    }
                }
            } finally {
                database.getNetworksReadLock().unlock();
            }
            System.out.println("Created " + stations.size() + " fake stations");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void doImportantStuff(StationDatabase database) {
        database.getNetworksReadLock().lock();
        try {
            for (Network n : database.getNetworks()) {
                for (Station s : n.getStations()) {
                    s.setNetwork(n);
                    for (Channel ch : s.getChannels()) {
                        ch.setStation(s);
                    }
                }
            }
        } finally {
            database.getNetworksReadLock().unlock();
        }
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> new EarthquakeSimulator().setVisible(true));
    }

    public ArrayList<SimulatedEarthquake> getEarthquakes() {
        return earthquakes;
    }

    public ArrayList<SimulatedStation> getStations() {
        return stations;
    }

    public FakeGlobalQuake getFakeGlobalQuake() {
        return fakeGlobalQuake;
    }

    public ClusterAnalysis getClusterAnalysis() {
        return clusterAnalysis;
    }

}

package globalquake.database;

import org.junit.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.Assert.*;

public class StationDatabaseManagerTest {

    @Test
    public void testConstructor(){
        StationDatabase stationDatabase = new StationDatabase();
        StationDatabaseManager databaseManager = new StationDatabaseManager(stationDatabase);

        assertEquals(stationDatabase, databaseManager.getStationDatabase());
    }

    @Test
    public void testAcceptChannel(){
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        var opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().size());
    }

    @Test
    public void testAcceptAnotherChannel(){
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        var opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().size());

        Channel dummyChannel2 = new Channel("coolChannel", "00", 50, 50, 0, 0, null);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel2);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().size());

        Channel channel = stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().stream().findAny().get();
        assertEquals(50, channel.getLatitude(), 1e-6);
    }

    @Test
    public void testRemoveStationSource(){
        StationSource stationSource1 = new StationSource("1", "");
        StationSource stationSource2 = new StationSource("2", "");
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource1);
        Channel dummyChannelDup = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource2);
        Channel dummyChannel2 = new Channel("coolChannel2", "00", 50, 0, 0, 0, stationSource2);

        StationDatabase stationDatabase = new StationDatabase();
        StationDatabaseManager stationDatabaseManager = new StationDatabaseManager(stationDatabase);

        stationDatabase.getStationSources().add(stationSource1);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannelDup);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        Optional<Station> opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());

        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel2);
        assertEquals(2, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());

        assertEquals(2, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(0).getStationSources().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(1).getStationSources().size());

        // REMOVE STATION SOURCE 1
        stationDatabaseManager.removeAllStationSources(List.of(stationSource1));

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());

        assertEquals(2, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(0).getStationSources().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(1).getStationSources().size());

        stationDatabaseManager.removeAllStationSources(List.of(stationSource2));

        assertEquals(0, stationDatabase.getNetworks().size());
    }

    @Test
    public void testSelectedChannel(){
        SeedlinkNetwork dummySeedlinkNetwork = new SeedlinkNetwork("dummy", "", 5);
        StationSource stationSource1 = new StationSource("1", "");
        StationSource stationSource2 = new StationSource("2", "");
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource1);
        Channel dummyChannelDup = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource2);
        Channel dummyChannel2 = new Channel("coolChannel2", "00", 50, 0, 0, 0, stationSource2);
        dummyChannel2.getSeedlinkNetworks().put(dummySeedlinkNetwork, 0L);
        dummyChannel.getSeedlinkNetworks().put(dummySeedlinkNetwork, 0L);

        StationDatabase stationDatabase = new StationDatabase();
        StationDatabaseManager stationDatabaseManager = new StationDatabaseManager(stationDatabase);

        stationDatabase.getStationSources().add(stationSource1);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannelDup);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        Optional<Station> opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());

        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel2);
        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());

        dummyStation.setSelectedChannel(dummyChannel2);
        assertEquals(dummyChannel2, dummyStation.getSelectedChannel());
        assertEquals(dummyStation, dummyStation);
        assertEquals(dummyStation, stationDatabase.getNetworks().get(0).getStations().get(0));
        assertEquals(dummyChannel2, stationDatabase.getNetworks().get(0).getStations().get(0).getSelectedChannel());

        assertEquals(2, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());

        assertEquals(2, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(0).getStationSources().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(1).getStationSources().size());

        // REMOVE STATION SOURCE 2
        stationDatabaseManager.removeAllStationSources(List.of(stationSource2));

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());

        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().size());
        assertEquals(dummyChannel, stationDatabase.getNetworks().get(0).getStations().get(0).getSelectedChannel());
    }


}
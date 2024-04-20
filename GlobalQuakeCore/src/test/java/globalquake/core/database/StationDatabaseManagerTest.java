package globalquake.core.database;

import gqserver.api.packets.station.InputType;
import org.junit.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.Assert.*;

@SuppressWarnings("OptionalGetWithoutIsPresent")
public class StationDatabaseManagerTest {

    @Test
    public void testConstructor() {
        StationDatabase stationDatabase = new StationDatabase();
        StationDatabaseManager databaseManager = new StationDatabaseManager(stationDatabase);

        assertEquals(stationDatabase, databaseManager.getStationDatabase());
    }

    @Test
    public void testAcceptChannel() {
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null, -1, InputType.UNKNOWN);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        var opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().size());
    }

    @Test
    public void testAcceptAnotherChannel() {
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null, -1, InputType.UNKNOWN);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);

        assertEquals(1, stationDatabase.getNetworks().size());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().size());
        var opt = stationDatabase.getNetworks().get(0).getStations().stream().findAny();
        assertTrue(opt.isPresent());
        assertEquals(1, stationDatabase.getNetworks().get(0).getStations().stream().findAny().get().getChannels().size());

        Channel dummyChannel2 = new Channel("coolChannel", "00", 50, 50, 0, 0, null, -1, InputType.UNKNOWN);
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
    public void testRemoveStationSource() {
        StationSource stationSource1 = new StationSource("1", "");
        StationSource stationSource2 = new StationSource("2", "");
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource1, -1, InputType.UNKNOWN);
        Channel dummyChannelDup = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource2, -1, InputType.UNKNOWN);
        Channel dummyChannel2 = new Channel("coolChannel2", "00", 50, 0, 0, 0, stationSource2, -1, InputType.UNKNOWN);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.getStationSources().add(stationSource1);
        stationDatabase.getStationSources().add(stationSource2);
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
    public void testBasicallyEverything() {
        SeedlinkNetwork dummySeedlinkNetwork = new SeedlinkNetwork("dummy", "", 5);
        SeedlinkNetwork dummySeedlinkNetwork2 = new SeedlinkNetwork("dummy2", "", 5);
        StationSource stationSource1 = new StationSource("1", "");
        StationSource stationSource2 = new StationSource("2", "");
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource1, -1, InputType.UNKNOWN);
        Channel dummyChannelDup = new Channel("coolChannel", "00", 50, 0, 0, 0, stationSource2, -1, InputType.UNKNOWN);
        Channel dummyChannel2 = new Channel("coolChannel2", "00", 50, 0, 0, 0, stationSource2, -1, InputType.UNKNOWN);
        dummyChannel2.getSeedlinkNetworks().put(dummySeedlinkNetwork, 1000L);
        dummySeedlinkNetwork2.selectedStations = 10;
        dummySeedlinkNetwork.selectedStations = 100;
        dummyChannel.getSeedlinkNetworks().put(dummySeedlinkNetwork, 500L);
        dummyChannel.getSeedlinkNetworks().put(dummySeedlinkNetwork2, 100L);
        assertEquals(dummySeedlinkNetwork2, dummyChannel.selectBestSeedlinkNetwork());

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.getStationSources().add(stationSource1);
        stationDatabase.getStationSources().add(stationSource2);
        stationDatabase.getSeedlinkNetworks().add(dummySeedlinkNetwork);
        stationDatabase.getSeedlinkNetworks().add(dummySeedlinkNetwork2);
        StationDatabaseManager stationDatabaseManager = new StationDatabaseManager(stationDatabase);

        stationDatabase.getStationSources().add(stationSource1);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannelDup);
        assertEquals(dummyChannel, stationDatabaseManager.getStationDatabase().getNetworks().get(0).getStations().get(0).getChannels().get(0));

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

        // TEST REMOVE SEEDLINK NETWORK
        assertTrue(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork));
        assertTrue(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork2));

        stationDatabaseManager.removeAllSeedlinks(List.of(dummySeedlinkNetwork));

        assertFalse(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork));
        assertTrue(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork2));

        stationDatabaseManager.removeAllSeedlinks(List.of(dummySeedlinkNetwork2));
        assertFalse(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork));
        assertFalse(dummyChannel.getSeedlinkNetworks().containsKey(dummySeedlinkNetwork2));
        assertNull(dummyChannel.selectBestSeedlinkNetwork());
    }

    @Test
    public void testChannelUpdate() {
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null, -1, InputType.UNKNOWN);
        Channel dummyChannelNew = new Channel("coolChannel", "00", 50, 50, 0, 0, null, -1, InputType.UNKNOWN);

        dummyNetwork.getStations().add(dummyStation);
        dummyStation.getChannels().add(dummyChannel);
        dummyStation.getChannels().add(dummyChannelNew);

        StationDatabase stationDatabase = new StationDatabase();
        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannel);

        dummyStation.setSelectedChannel(dummyChannel);
        assertEquals(dummyChannel, stationDatabase.getNetworks().get(0).getStations().get(0).getSelectedChannel());
        assertTrue(stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().contains(stationDatabase.getNetworks().get(0).getStations().get(0).getSelectedChannel()));

        stationDatabase.acceptChannel(dummyNetwork, dummyStation, dummyChannelNew);
        assertEquals(50, stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().get(0).getLatitude(), 1e-6);
        assertTrue(stationDatabase.getNetworks().get(0).getStations().get(0).getChannels().contains(stationDatabase.getNetworks().get(0).getStations().get(0).getSelectedChannel()));
    }

    @Test
    public void testAcceptNetworks() {
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null, -1, InputType.UNKNOWN);
        Channel dummyChannelNew = new Channel("coolChannel", "00", 50, 50, 0, 0, null, -1, InputType.UNKNOWN);

        dummyNetwork.getStations().add(dummyStation);
        dummyStation.getChannels().add(dummyChannel);
        dummyStation.getChannels().add(dummyChannelNew);

        StationDatabaseManager databaseManager = new StationDatabaseManager(new StationDatabase());
        databaseManager.acceptNetworks(List.of(dummyNetwork));
    }

    @Test
    public void testChannelDeselectWhenRemoveAllSeedlinks() {
        SeedlinkNetwork seedlinkNetwork = new SeedlinkNetwork("dummy", "D", 5);
        Network dummyNetwork = new Network("coolNetwork", "");
        Station dummyStation = new Station(dummyNetwork, "coolStation", "", 0, 0, 0);
        Channel dummyChannel = new Channel("coolChannel", "00", 50, 0, 0, 0, null, -1, InputType.UNKNOWN);
        Channel dummyChannelNew = new Channel("coolChannel", "00", 50, 50, 0, 0, null, -1, InputType.UNKNOWN);

        dummyNetwork.getStations().add(dummyStation);
        dummyStation.getChannels().add(dummyChannel);
        dummyStation.getChannels().add(dummyChannelNew);

        StationDatabaseManager databaseManager = new StationDatabaseManager(new StationDatabase());
        databaseManager.acceptNetworks(List.of(dummyNetwork));

        dummyChannel.getSeedlinkNetworks().put(seedlinkNetwork, 0L);
        dummyStation.setSelectedChannel(dummyChannel);

        databaseManager.removeAllSeedlinks(List.of(seedlinkNetwork));
        assertTrue(dummyChannel.getSeedlinkNetworks().isEmpty());
        assertNull(dummyStation.getSelectedChannel());
    }


}
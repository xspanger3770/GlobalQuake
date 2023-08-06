package globalquake.database;

import globalquake.exception.FatalIOException;
import globalquake.main.Main;

import java.io.*;
import java.time.LocalDateTime;
import java.util.List;

public class StationDatabaseManager {

    private static final File STATIONS_FOLDER = new File(Main.MAIN_FOLDER, "/stationDatabase/");
    private StationDatabase stationDatabase;

    public void load() throws FatalIOException{
        File file = getDatabaseFile();
        if (!file.getParentFile().exists()) {
            if(!file.getParentFile().mkdirs()){
               throw new FatalIOException("Unable to create database file directory!", null);
            }
        }

        if(file.exists()){
            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
                stationDatabase = (StationDatabase) in.readObject();
                in.close();

                System.out.println("Load successfull");
            } catch (ClassNotFoundException | IOException e) {
                Main.getErrorHandler().handleException(new FatalIOException("Unable to read station database!", e));
            }
        }

        if(stationDatabase == null){
            System.out.println("A new database created");
            stationDatabase = new StationDatabase();
        }
    }

    public void save() throws FatalIOException{
        File file = getDatabaseFile();
        if (!file.getParentFile().exists()) {
            if(!file.getParentFile().mkdirs()){
                throw new FatalIOException("Unable to create database file directory!", null);
            }
        }

        if(stationDatabase == null){
            return;
        }

        stationDatabase.getDatabaseReadLock().lock();
        try{
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
            out.writeObject(stationDatabase);
            out.close();
        }catch(IOException e){
            throw new FatalIOException("Unable to save station database!", e);
        } finally {
            stationDatabase.getDatabaseReadLock().unlock();
        }
    }

    public void runUpdate(List<StationSource> toBeUpdated, Runnable onFinish) {
        new Thread(() -> {
            toBeUpdated.parallelStream().forEach(stationSource -> {
                stationSource.getStatus().setString("Updating...");
                try {
                    List<Network> networkList = FDSNWSDownloader.downloadFDSNWS(stationSource);
                    stationSource.getStatus().setString("Updating database...");
                    StationDatabaseManager.this.acceptNetworks(networkList);
                    stationSource.getStatus().setString("Done");
                    stationSource.getStatus().setValue(100);
                    stationSource.setLastUpdate(LocalDateTime.now());
                } catch (Exception e) {
                    stationSource.getStatus().setString("Error!");
                    stationSource.getStatus().setValue(0);
                }
            });
            if (onFinish != null) {
                onFinish.run();
            }
        }).start();
    }

    private void acceptNetworks(List<Network> networkList) {
        stationDatabase.getDatabaseWriteLock().lock();
        try{
            for(Network network:networkList){
                for(Station station: network.getStations()){
                    for(Channel channel:station.getChannels()){
                        stationDatabase.getOrCreateChannel(network, station, channel);
                    }
                }
            }
        }finally {
            System.out.println(stationDatabase.getNetworks().size());
            stationDatabase.getDatabaseWriteLock().unlock();
        }
    }

    private static File getDatabaseFile() {
        return new File(STATIONS_FOLDER, "database.dat");
    }

    public StationDatabase getStationDatabase() {
        return stationDatabase;
    }

    public void runAvailabilityCheck(List<SeedlinkNetwork> toBeUpdated, Runnable onFinish) {
        new Thread(() -> {
            toBeUpdated.parallelStream().forEach(seedlinkNetwork -> {
                        try {
                            SeedlinkCommunicator.runAvailabilityCheck(seedlinkNetwork, stationDatabase);
                        } catch (Exception e) {
                            seedlinkNetwork.getStatus().setString("Error!");
                            seedlinkNetwork.getStatus().setValue(0);
                        }
                    }
            );
            if (onFinish != null) {
                onFinish.run();
            }
        }).start();
    }
}

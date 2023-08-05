package globalquake.database;

import globalquake.exception.FatalIOException;
import globalquake.main.Main;

import java.io.*;
import java.time.LocalDateTime;
import java.util.List;

public class StationDatabaseManager {

    private static final File STATIONS_FOLDER = new File(Main.MAIN_FOLDER, "/stationDatabase/");;
    private StationDatabase stationDatabase;

    public StationDatabaseManager() {
    }

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
        try{
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
            out.writeObject(stationDatabase);
            out.close();
        }catch(IOException e){
            throw new FatalIOException("Unable to save station database!", e);
        }
    }

    public void runUpdate(List<StationSource> stationSources) {
        stationSources.parallelStream().forEach(stationSource -> {
            stationSource.getStatus().setString("Updating...");
            stationSource.setLastUpdate(LocalDateTime.now());
        });
    }

    private static File getDatabaseFile() {
        return new File(STATIONS_FOLDER, "database.dat");
    }

    public StationDatabase getStationDatabase() {
        return stationDatabase;
    }
}

package globalquake.utils;

import java.io.*;
import java.net.URL;
import java.util.HashMap;
import java.util.List;

import globalquake.core.regions.Regions;

public class LookupTableIO {
    public static boolean exportLookupTableToFile() {
        List<HashMap<String, Double>> lookupTables = Regions.generateLookupTablesInParallel();
        HashMap<String, Double> lookupTable = new HashMap<>();
        for (HashMap<String, Double> singleLT : lookupTables) {
            lookupTable.putAll(singleLT);
        }

        return performExport(lookupTable);
    }

    public static boolean exportLookupTableToFile(HashMap<String, Double> lookupTable) {
        return performExport(lookupTable);
    }

    private static boolean performExport(HashMap<String, Double> lookupTable) {

        try {
            ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream("lookupTable.dat"));
            output.writeObject(lookupTable);
            output.close();
        } catch (Exception e) {
            System.err.println("Unable to save a lookup table! " + e);
            return false;
        }

        return true;
    }

    @SuppressWarnings("unchecked")
    public static HashMap<String, Double> importLookupTableFromFile() throws IOException {
        String path = "lookup/lookupTable.dat";
        URL resource = ClassLoader.getSystemClassLoader().getResource(path);

        if (resource == null) {
            System.err.printf("Unable to load a lookup table: %s", path);
            return null;
        }

        HashMap<String, Double> lookupTable;
        try {
            ObjectInput input = new ObjectInputStream(resource.openStream());
            lookupTable = (HashMap<String, Double>) input.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new IOException("Unable to load stream of a lookup table! ", e);
        }

        return lookupTable;
    }
}

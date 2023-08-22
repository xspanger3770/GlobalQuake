package globalquake.database;

import edu.sc.seis.seisFile.seedlink.SeedlinkReader;
import org.tinylog.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.TimeZone;

public class SeedlinkCommunicator {

    public static final long UNKNOWN_DELAY = Long.MIN_VALUE;
    private static final SimpleDateFormat FORMAT_UTC_SHORT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private static final SimpleDateFormat FORMAT_UTC_LONG = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss.SSSS");
    private static final long MAX_DELAY_MS = 1000 * 60 * 60 * 24L;
    public static final int SEEDLINK_TIMEOUT_SECONDS = 10;

    static{
        FORMAT_UTC_SHORT.setTimeZone(TimeZone.getTimeZone("UTC"));
        FORMAT_UTC_LONG.setTimeZone(TimeZone.getTimeZone("UTC"));
    }

    public static void runAvailabilityCheck(SeedlinkNetwork seedlinkNetwork, StationDatabase stationDatabase) throws Exception {
        seedlinkNetwork.getStatus().setString("Connecting...");
        seedlinkNetwork.getStatus().setValue(0);
        SeedlinkReader reader = new SeedlinkReader(seedlinkNetwork.getHost(), seedlinkNetwork.getPort(), SEEDLINK_TIMEOUT_SECONDS, false);

        seedlinkNetwork.getStatus().setString("Downloading...");
        seedlinkNetwork.getStatus().setValue(33);
        String infoString = reader.getInfoString(SeedlinkReader.INFO_STREAMS).trim().replaceAll("[^\\u0009\\u000a\\u000d\\u0020-\\uD7FF\\uE000-\\uFFFD]", " ");

        seedlinkNetwork.getStatus().setString("Parsing...");
        seedlinkNetwork.getStatus().setValue(66);
        parseAvailability(infoString, stationDatabase, seedlinkNetwork);

        reader.close();
        seedlinkNetwork.getStatus().setString("Done");
        seedlinkNetwork.getStatus().setValue(100);
    }

    private static void parseAvailability(String infoString, StationDatabase stationDatabase, SeedlinkNetwork seedlinkNetwork) throws Exception {
        seedlinkNetwork.availableStations = 0;

        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        DocumentBuilder db = dbf.newDocumentBuilder();
        Document doc = db.parse(new InputSource(new StringReader(infoString)));
        doc.getDocumentElement().normalize();
        NodeList nodeList = doc.getElementsByTagName("station");
        System.out.println("Found " + nodeList.getLength() + " available stations.");
        for (int itr = 0; itr < nodeList.getLength(); itr++) {
            Node node = nodeList.item(itr);
            String stationCode = node.getAttributes().getNamedItem("name").getTextContent();
            String networkCode = node.getAttributes().getNamedItem("network").getTextContent();
            NodeList channelList = ((Element) node).getElementsByTagName("stream");
            for (int k = 0; k < channelList.getLength(); k++) {
                Node channel = channelList.item(k);
                String locationCode = channel.getAttributes().getNamedItem("location").getTextContent();
                String channelName = channel.getAttributes().getNamedItem("seedname").getTextContent();
                String endDate = channel.getAttributes().getNamedItem("end_time").getTextContent();

                long delay = UNKNOWN_DELAY;

                try {
                    Calendar end = Calendar.getInstance();
                    end.setTime(endDate.contains("-") ? FORMAT_UTC_SHORT.parse(endDate) : FORMAT_UTC_LONG.parse(endDate));

                    delay = System.currentTimeMillis() - end.getTimeInMillis();

                    if (delay > MAX_DELAY_MS) {
                        continue;
                    }

                } catch(NumberFormatException e){
                    Logger.warn(new RuntimeException("Failed to get delay from %s, %s: %s".formatted(stationCode, seedlinkNetwork.getHost(), e.getMessage())));
                }

                addAvailableChannel(networkCode, stationCode, channelName, locationCode, delay, seedlinkNetwork, stationDatabase);
            }
        }
    }

    private static void addAvailableChannel(String networkCode, String stationCode, String channelName, String locationCode, long delay, SeedlinkNetwork seedlinkNetwork, StationDatabase stationDatabase) {
        stationDatabase.getDatabaseWriteLock().lock();
        try {
            Channel channel = StationDatabase.getChannel(stationDatabase.getNetworks(), networkCode, stationCode, channelName, locationCode);
            if (channel == null) {
                return;
            }

            seedlinkNetwork.availableStations++;
            channel.getSeedlinkNetworks().put(seedlinkNetwork, delay);
        }finally {
            stationDatabase.getDatabaseWriteLock().unlock();
        }
    }

}

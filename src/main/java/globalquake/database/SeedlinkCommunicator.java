package globalquake.database;

import edu.sc.seis.seisFile.seedlink.SeedlinkReader;
import globalquake.utils.TimeFixer;
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

public class SeedlinkCommunicator {

    private static final SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private static final SimpleDateFormat format3 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss.SSSS");

    public static void runAvailabilityCheck(SeedlinkNetwork seedlinkNetwork, StationDatabase stationDatabase) throws Exception {
        seedlinkNetwork.getStatus().setString("Connecting...");
        seedlinkNetwork.getStatus().setValue(0);
        SeedlinkReader reader = new SeedlinkReader(seedlinkNetwork.getHost(), seedlinkNetwork.getPort(), 10, false);

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

    private static void parseAvailability(String infoString, StationDatabase stationDatabase, SeedlinkNetwork seedlinkNetwork) throws Exception{
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
                Calendar end = Calendar.getInstance();
                end.setTime(endDate.contains("-") ? format2.parse(endDate) : format3.parse(endDate));
                long delay = System.currentTimeMillis() - end.getTimeInMillis() - TimeFixer.offset();
                if (delay > 1000 * 60 * 60 * 24 * 7) {
                    continue;
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

            channel.getSeedlinkNetworks().add(seedlinkNetwork);
        }finally {
            stationDatabase.getDatabaseWriteLock().unlock();
        }
    }

}

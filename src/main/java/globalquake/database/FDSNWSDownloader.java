package globalquake.database;

import globalquake.exception.FdnwsDownloadException;
import globalquake.main.Main;
import org.tinylog.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXParseException;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class FDSNWSDownloader {

    private static final String CHANNELS = "EHZ,SHZ,HHZ,BHZ";
    private static final SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");
    private static final int TIMEOUT_SECONDS = 20;

    public static List<Network> downloadFDSNWS(StationSource stationSource) throws Exception {
        List<Network> result = new ArrayList<>();
        downloadFDSNWS(stationSource, result, -180, 180);
        return result;
    }

    public static void downloadFDSNWS(StationSource stationSource, List<Network> result, double minLon, double maxLon) throws Exception {
        URL url = new URL("%squery?minlongitude=%s&maxlongitude=%s&level=channel&endafter=%s&format=xml&channel=%s".formatted(stationSource.getUrl(), minLon, maxLon, format1.format(new Date()), CHANNELS));

        System.out.println("Connecting to " + url);

        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setConnectTimeout(TIMEOUT_SECONDS * 1000);
        con.setReadTimeout(TIMEOUT_SECONDS * 1000);

        int response = con.getResponseCode();

        if (response == 413) {
            System.err.println("413! Splitting...");
            if(maxLon - minLon < 0.1){
                System.err.println("This can't go forewer");
                return;
            }

            downloadFDSNWS(stationSource, result, minLon, (minLon + maxLon) / 2.0);
            downloadFDSNWS(stationSource, result, (minLon + maxLon) / 2.0, maxLon);
        } else if(response / 100 == 2) {
            InputStream inp = con.getInputStream();
            downloadFDSNWS(stationSource, result, inp);
        } else {
            throw new FdnwsDownloadException("HTTP Status %d!".formatted(response));
        }
    }

    private static void downloadFDSNWS(StationSource stationSource, List<Network> result, InputStream inp) throws Exception {
        DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
        f.setNamespaceAware(false);
        f.setValidating(false);
        final CountInputStream in = new CountInputStream(inp);

        in.setEvent(() ->  stationSource.getStatus().setString("Downloading %dkB".formatted(in.getCount() / 1024)));

        Document doc;
        try {
            doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);
        }catch(SAXParseException e){
            Logger.error(e);
            return;
        }

        doc.getDocumentElement().normalize();

        Element root = doc.getDocumentElement();
        parseNetworks(result, stationSource, root);
    }

    private static void parseNetworks(List<Network> result, StationSource stationSource, Element root) {
        NodeList networks = root.getElementsByTagName("Network");
        for (int i = 0; i < networks.getLength(); i++) {
            try {
                String networkCode = obtainAttribute(networks.item(i), "code", "unknown");
                if (networkCode.equalsIgnoreCase("unknown")) {
                    System.err.println("ERR: no network code wtf.");
                    continue;
                }
                String networkDescription = obtainElement(networks.item(i), "Description", "");
                parseStations(result, stationSource, networks, i, networkCode, networkDescription);
            } catch (Exception e) {
                Main.getErrorHandler().handleException(e);
            }
        }
    }

    private static void parseStations(List<Network> result, StationSource stationSource, NodeList networks, int i, String networkCode, String networkDescription) {
        NodeList stations = ((Element) networks.item(i)).getElementsByTagName("Station");
        for (int j = 0; j < stations.getLength(); j++) {
            Node stationNode = stations.item(j);
            String stationCode = stationNode.getAttributes().getNamedItem("code").getNodeValue();
            String stationSite = ((Element) stationNode).getElementsByTagName("Site").item(0).getTextContent();
            // todo station-specific lat lon alt
            parseChannels(result, stationSource, networkCode, networkDescription, (Element) stationNode, stationCode, stationSite);
        }
    }

    private static void parseChannels(List<Network> result, StationSource stationSource, String networkCode, String networkDescription, Element stationNode, String stationCode, String stationSite) {
        NodeList channels = stationNode.getElementsByTagName("Channel");
        for (int k = 0; k < channels.getLength(); k++) {
                // Necessary values: lat lon alt sampleRate, Other can fail

            Node channelNode = channels.item(k);
            String channel = channelNode.getAttributes().getNamedItem("code").getNodeValue();
            String locationCode = channelNode.getAttributes().getNamedItem("locationCode")
                    .getNodeValue();
            double lat = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Latitude").item(0).getTextContent());
            double lon = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Longitude").item(0).getTextContent());
            double alt = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Elevation").item(0).getTextContent());
            double sampleRate = Double.parseDouble(((Element) channelNode)
                    .getElementsByTagName("SampleRate").item(0).getTextContent());

            addChannel(result, stationSource, networkCode, networkDescription, stationCode, stationSite, channel,
                    locationCode, lat, lon, alt, sampleRate);
        }
    }

    private static void addChannel(List<Network> result, StationSource stationSource, String networkCode, String networkDescription, String stationCode, String stationSite, String channelCode, String locationCode, double lat, double lon, double alt, double sampleRate) {
        Network network = StationDatabase.getOrCreateNetwork(result, networkCode, networkDescription);
        Station station = StationDatabase.getOrCreateStation(network, stationCode, stationSite, lat, lon, alt);
        StationDatabase.getOrCreateChannel(station, channelCode, locationCode, lat, lon, alt, sampleRate, stationSource);
    }

    public static String obtainElement(Node item, String name, String defaultValue) {
        try {
            return ((Element) item).getElementsByTagName(name).item(0).getTextContent();
        } catch (Exception e) {
            return defaultValue;
        }
    }

    public static String obtainAttribute(Node item, String name, String defaultValue) {
        try {
            return item.getAttributes().getNamedItem(name).getNodeValue();
        } catch (Exception e) {
            return defaultValue;
        }
    }

}

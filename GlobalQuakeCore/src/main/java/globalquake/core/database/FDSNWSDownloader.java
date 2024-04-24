package globalquake.core.database;

import globalquake.core.GlobalQuake;
import globalquake.core.exception.FdnwsDownloadException;
import gqserver.api.packets.station.InputType;
import org.tinylog.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import javax.net.ssl.*;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.*;
import java.math.BigDecimal;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class FDSNWSDownloader {

    private static final DateTimeFormatter format1 = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss").withZone(ZoneId.systemDefault());
    private static final int TIMEOUT_SECONDS = 120;

    public static final List<Character> SUPPORTED_BANDS = List.of('E', 'S', 'H', 'B', 'C', 'A');
    public static final List<Character> SUPPORTED_INSTRUMENTS = List.of('H', 'L', 'G', 'M', 'N', 'C');
    private static final List<SensitivityCorrection> sensitivityCorrections;

    static {
        TrustManager[] trustAllCerts = new TrustManager[]{
                new X509TrustManager() {
                    public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                        return null;
                    }

                    public void checkClientTrusted(
                            java.security.cert.X509Certificate[] certs, String authType) {
                    }

                    public void checkServerTrusted(
                            java.security.cert.X509Certificate[] certs, String authType) {
                    }
                }
        };
        // Install the all-trusting trust manager
        try {
            SSLContext sc = SSLContext.getInstance("TLS");
            sc.init(null, trustAllCerts, new java.security.SecureRandom());
            HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());
        } catch (Exception e) {
            Logger.error(e);
        }

        sensitivityCorrections = new ArrayList<>();
        try {
            File file = new File(GlobalQuake.mainFolder, "sensitivity_corrections.txt");
            if (!file.exists()) {
                if (!file.createNewFile()) {
                    throw new RuntimeException("Failed to create sensitivity_corrections.txt file!");
                }
            }
            sensitivityCorrections.addAll(loadSensitivityCorrections(file.getAbsolutePath()));
        } catch (Exception e) {
            Logger.error("Unable to load sensitivity corrections", e);
        }
    }

    public static List<SensitivityCorrection> loadSensitivityCorrections(String filePath) throws IOException {
        List<SensitivityCorrection> corrections = new ArrayList<>();

        Logger.info("Loading sensitivity corrections...");

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] fields = line.split(";");
                if (fields.length == 3) {
                    String networkCode = fields[0].trim();
                    String stationCode = fields[1].trim();
                    double multiplier = Double.parseDouble(fields[2].trim());
                    SensitivityCorrection correction = new SensitivityCorrection(networkCode, stationCode, multiplier);
                    corrections.add(correction);
                } else {
                    Logger.error("Invalid line: " + line);
                }
            }
        }

        Logger.info("Loaded %d sensitivity correction rules.");

        return corrections;
    }

    public static void main(String[] args) throws Exception {
        GlobalQuake.prepare(new File("./.GlobalQuakeData/"), null);
        var a = loadSensitivityCorrections(new File(GlobalQuake.mainFolder, "sensitivity_corrections.txt").getAbsolutePath());
        System.err.println(a);
    }

    private static List<String> downloadWadl(StationSource stationSource) throws Exception {
        URL url = new URL("%sapplication.wadl".formatted(stationSource.getUrl()));


        URLConnection con = url.openConnection();

        if (con instanceof HttpsURLConnection httpsURLConnection) {
            httpsURLConnection.setHostnameVerifier(getHostnameVerifier());
        }

        con.setConnectTimeout(TIMEOUT_SECONDS * 1000);
        con.setReadTimeout(TIMEOUT_SECONDS * 1000);
        InputStream inp = con.getInputStream();

        Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(inp);
        doc.getDocumentElement().normalize();

        List<String> paramNames = new ArrayList<>();
        NodeList paramNodes = doc.getElementsByTagName("param");
        for (int i = 0; i < paramNodes.getLength(); i++) {
            Node paramNode = paramNodes.item(i);
            if (paramNode.getNodeType() == Node.ELEMENT_NODE) {
                Element paramElement = (Element) paramNode;
                String paramName = paramElement.getAttribute("name");
                paramNames.add(paramName);
            }
        }

        return paramNames;
    }

    private static HostnameVerifier getHostnameVerifier() {
        return (hostname, session) -> true;
    }

    public static List<Network> downloadFDSNWS(StationSource stationSource, String addons) throws Exception {
        List<Network> result = new ArrayList<>();
        downloadFDSNWS(stationSource, result, -180, 180, addons);
        Logger.info("%d Networks downloaded".formatted(result.size()));
        return result;
    }

    public static void downloadFDSNWS(StationSource stationSource, List<Network> result, double minLon, double maxLon, String addons) throws Exception {
        List<String> supportedAttributes = downloadWadl(stationSource);
        URL url;

        StringBuilder addonsResult = new StringBuilder();
        List<String> addonsSplit = List.of(addons.split("&"));
        for (String str : addonsSplit) {
            if (str.isEmpty()) {
                continue;
            }
            if (supportedAttributes.contains(str.split("=")[0])) {
                addonsResult.append("&");
                addonsResult.append(str);
            } else {
                Logger.warn("Addon not supported: %s".formatted(str.split("=")[0]));
            }
        }

        if (supportedAttributes.contains("endafter") && addons.isEmpty()) {
            url = new URL("%squery?minlongitude=%s&maxlongitude=%s&level=channel&endafter=%s&format=xml&channel=??Z%s".formatted(stationSource.getUrl(), minLon, maxLon, format1.format(Instant.now()), addonsResult));
        } else {
            url = new URL("%squery?minlongitude=%s&maxlongitude=%s&level=channel&format=xml&channel=??Z%s".formatted(stationSource.getUrl(), minLon, maxLon, addonsResult));
        }


        Logger.info("Connecting to " + url);

        URLConnection con = url.openConnection();
        int response = -1;

        if (con instanceof HttpsURLConnection httpsURLConnection) {
            httpsURLConnection.setHostnameVerifier(getHostnameVerifier());
            response = httpsURLConnection.getResponseCode();
        } else if (con instanceof HttpURLConnection httpURLConnection) {
            response = httpURLConnection.getResponseCode();
        }
        con.setConnectTimeout(TIMEOUT_SECONDS * 1000);
        con.setReadTimeout(TIMEOUT_SECONDS * 1000);

        if (response == 413) {
            Logger.debug("413! Splitting...");
            stationSource.getStatus().setString("Splitting...");
            if (maxLon - minLon < 0.1) {
                return;
            }

            downloadFDSNWS(stationSource, result, minLon, (minLon + maxLon) / 2.0, addons);
            downloadFDSNWS(stationSource, result, (minLon + maxLon) / 2.0, maxLon, addons);
        } else if (response / 100 == 2) {
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

        in.setEvent(() -> stationSource.getStatus().setString("Downloading %dkB".formatted(in.getCount() / 1024)));

        String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);

        // some FDSNWS providers send empty document if no stations found by given parameters
        if (text.isEmpty()) {
            return;
        }

        Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(new InputSource(new StringReader(text)));

        doc.getDocumentElement().normalize();

        Element root = doc.getDocumentElement();
        parseNetworks(result, stationSource, root);
    }

    private static void parseNetworks(List<Network> result, StationSource stationSource, Element root) {
        NodeList networks = root.getElementsByTagName("Network");
        for (int i = 0; i < networks.getLength(); i++) {
            String networkCode = obtainAttribute(networks.item(i), "code", "unknown");
            if (networkCode.equalsIgnoreCase("unknown")) {
                Logger.debug("ERR: no network code wtf.");
                continue;
            }
            String networkDescription = obtainElement(networks.item(i), "Description", "");
            parseStations(result, stationSource, networks, i, networkCode, networkDescription);
        }
    }

    private static void parseStations(List<Network> result, StationSource stationSource, NodeList networks, int i, String networkCode, String networkDescription) {
        NodeList stations = ((Element) networks.item(i)).getElementsByTagName("Station");
        for (int j = 0; j < stations.getLength(); j++) {
            Node stationNode = stations.item(j);
            String stationCode = stationNode.getAttributes().getNamedItem("code").getNodeValue();
            String stationSite = ((Element) stationNode).getElementsByTagName("Site").item(0).getTextContent();

            double lat = Double.parseDouble(
                    ((Element) stationNode).getElementsByTagName("Latitude").item(0).getTextContent());
            double lon = Double.parseDouble(
                    ((Element) stationNode).getElementsByTagName("Longitude").item(0).getTextContent());
            double alt = Double.parseDouble(
                    ((Element) stationNode).getElementsByTagName("Elevation").item(0).getTextContent());

            parseChannels(result, stationSource, networkCode, networkDescription, (Element) stationNode, stationCode, stationSite, lat, lon, alt);
        }
    }

    private static boolean isWithinDateRange(String startDateStr, String endDateStr) {
        // Try parsing with 'Z' and without 'Z'
        SimpleDateFormat dateFormatWithZ = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
        SimpleDateFormat dateFormatWithoutZ = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");

        try {
            Date startDate = startDateStr != null ?
                    parseDate(startDateStr, dateFormatWithZ, dateFormatWithoutZ) : null;
            Date currentDate = new Date();

            if (endDateStr != null) {
                Date endDate = parseDate(endDateStr, dateFormatWithZ, dateFormatWithoutZ);
                // Check if the current date is within the start and end dates
                return (startDate == null || currentDate.after(startDate)) && currentDate.before(endDate);
            } else {
                // If there is no end date, check if the current date is after the start date
                return (startDate == null || currentDate.after(startDate));
            }
        } catch (ParseException e) {
            Logger.error(e);
            return false;
        }
    }

    private static Date parseDate(String dateString, SimpleDateFormat... dateFormats) throws ParseException {
        for (SimpleDateFormat dateFormat : dateFormats) {
            try {
                return dateFormat.parse(dateString);
            } catch (ParseException ignored) {
                // Try the next format if the current one fails
            }
        }
        // If none of the formats match, throw an exception
        throw new ParseException("Unparseable date: " + dateString, 0);
    }

    private static void parseChannels(
            List<Network> result, StationSource stationSource, String networkCode, String networkDescription,
            Element stationNode, String stationCode, String stationSite,
            double stationLat, double stationLon, double stationAlt) {
        NodeList channels = stationNode.getElementsByTagName("Channel");
        for (int k = 0; k < channels.getLength(); k++) {
            // Necessary values: lat lon alt sampleRate, Other can fail

            Node channelNode = channels.item(k);
            String channel = channelNode.getAttributes().getNamedItem("code").getNodeValue();

            String startDateStr = channelNode.getAttributes().getNamedItem("startDate") != null ?
                    channelNode.getAttributes().getNamedItem("startDate").getNodeValue() : null;

            // Get the "endDate" attribute if available
            String endDateStr = channelNode.getAttributes().getNamedItem("endDate") != null ?
                    channelNode.getAttributes().getNamedItem("endDate").getNodeValue() : null;

            if (!isWithinDateRange(startDateStr, endDateStr)) {
                continue;
            }

            String locationCode = channelNode.getAttributes().getNamedItem("locationCode")
                    .getNodeValue();
            double lat = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Latitude").item(0).getTextContent());
            double lon = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Longitude").item(0).getTextContent());
            double alt = Double.parseDouble(
                    ((Element) channelNode).getElementsByTagName("Elevation").item(0).getTextContent());

            double sensitivity = -1;
            InputType inputType = InputType.UNKNOWN;
            try {
                sensitivity = new BigDecimal(((Element) ((Element) (channelNode.getChildNodes()))
                        .getElementsByTagName("InstrumentSensitivity").item(0))
                        .getElementsByTagName("Value").item(0).getTextContent()).doubleValue();

                String inputUnits = ((Element) ((Element) ((Element) (channelNode.getChildNodes()))
                        .getElementsByTagName("InstrumentSensitivity").item(0))
                        .getElementsByTagName("InputUnits").item(0)).getElementsByTagName("Name").item(0).getTextContent();

                sensitivity *= getInputUnitsMultiplier(inputUnits);
                inputType = getInputType(inputUnits);
            } catch (NullPointerException e) {
                Logger.debug(
                        "No Sensitivity!!!! " + stationCode + " " + networkCode + " " + channel + " @ " + stationSource.getUrl());
            }

            var item = ((Element) channelNode)
                    .getElementsByTagName("SampleRate").item(0);

            // sample rate is not actually required as it is provided by the seedlink protocol itself
            double sampleRate = -1;
            if (item != null) {
                sampleRate = Double.parseDouble(((Element) channelNode)
                        .getElementsByTagName("SampleRate").item(0).getTextContent());
            }

            if (!isSupported(channel)) {
                continue;
            }

            addChannel(result, stationSource, networkCode, networkDescription, stationCode, stationSite, channel,
                    locationCode, lat, lon, alt, sampleRate, stationLat, stationLon, stationAlt, sensitivity, inputType);
        }
    }

    public static double getSensitivityCorrection(String networkCode, String stationCode) {
        for (SensitivityCorrection sensitivityCorrection : sensitivityCorrections) {
            if (sensitivityCorrection.match(networkCode, stationCode)) {
                return sensitivityCorrection.getMultiplier();
            }
        }
        return 1.0;
    }

    private static final Set<String> unknownUnits = new HashSet<>();

    private static final Map<String, InputType> unitTypeMap = new HashMap<>();
    private static final Map<String, Double> unitMultiplierMap = new HashMap<>();

    static {
        // Unit to InputType mapping
        unitTypeMap.put("m", InputType.DISPLACEMENT);
        unitTypeMap.put("nm", InputType.DISPLACEMENT);
        unitTypeMap.put("mm", InputType.DISPLACEMENT);

        unitTypeMap.put("m/s", InputType.VELOCITY);
        unitTypeMap.put("nm/s", InputType.VELOCITY);
        unitTypeMap.put("mm/s", InputType.VELOCITY);

        unitTypeMap.put("1m/s**2", InputType.ACCELERATION);
        unitTypeMap.put("m/s**2", InputType.ACCELERATION);
        unitTypeMap.put("nm/s**2", InputType.ACCELERATION);
        unitTypeMap.put("mm/s**2", InputType.ACCELERATION);

        // Unit to Multiplier mapping
        unitMultiplierMap.put("nm", 1E9);
        unitMultiplierMap.put("nm/s", 1E9);
        unitMultiplierMap.put("nm/s**2", 1E9);

        unitMultiplierMap.put("mm", 1E3);
        unitMultiplierMap.put("mm/s", 1E3);
        unitMultiplierMap.put("mm/s**2", 1E3);

        // other unidentified units: [volts, , m/s/s, counts, nt, none.specified, g, count, m/m, none, radians, rad/s, 1m/s**2, rad/sec, t, v, volt, r/s, kpa]
    }

    private static InputType getInputType(String inputUnits) {
        InputType inputType = unitTypeMap.getOrDefault(inputUnits.toLowerCase(), InputType.UNKNOWN);

        if (inputType == InputType.UNKNOWN) {
            unknownUnits.add(inputUnits.toLowerCase());
            Logger.debug("Unknown input units: %s".formatted(Arrays.toString(unknownUnits.toArray())));
        }

        return inputType;
    }

    private static double getInputUnitsMultiplier(String inputUnits) {
        return unitMultiplierMap.getOrDefault(inputUnits.toLowerCase(), 1.0);
    }

    private static boolean isSupported(String channel) {
        char band = channel.charAt(0);
        char instrument = channel.charAt(1);

        if (!(SUPPORTED_BANDS.contains(band))) {
            return false;
        }

        return SUPPORTED_INSTRUMENTS.contains(instrument);
    }

    private static void addChannel(
            List<Network> result, StationSource stationSource, String networkCode, String networkDescription,
            String stationCode, String stationSite, String channelCode, String locationCode,
            double lat, double lon, double alt, double sampleRate,
            double stationLat, double stationLon, double stationAlt, double sensitivity, InputType inputType) {
        Network network = StationDatabase.getOrCreateNetwork(result, networkCode, networkDescription);
        Station station = StationDatabase.getOrCreateStation(network, stationCode, stationSite, stationLat, stationLon, stationAlt);
        StationDatabase.getOrCreateChannel(station, channelCode, locationCode, lat, lon, alt, sampleRate, stationSource, sensitivity, inputType);
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

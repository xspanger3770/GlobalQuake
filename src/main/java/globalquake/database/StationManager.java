package globalquake.database;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.math.BigDecimal;
import java.net.URL;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;

import javax.swing.JOptionPane;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.SelectedStation;
import com.morce.globalquake.database.Station;
import com.morce.globalquake.database.StationDatabase;

import edu.sc.seis.seisFile.seedlink.SeedlinkReader;
import globalquake.main.Main;
import globalquake.utils.TimeFixer;

public class StationManager implements IStationManager {

	public static final ArrayList<DataSource> sources = new ArrayList<DataSource>();
	public static final ArrayList<SeedlinkNetwork> seedlinks = new ArrayList<SeedlinkNetwork>();
	
	private static final SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");
	private static final SimpleDateFormat format2 = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	private static final SimpleDateFormat format3 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss.SSSS");

	private static final String channels = "EHZ,SHZ,HHZ,BHZ";

	public static final String GEOFON = "geofon.gfz-potsdam.de";
	public static final String RESIF = "rtserve.resif.fr";
	public static final String IRIS_RTSERVER = "rtserve.iris.washington.edu";

	public static final int DATABASE_VERSION = 4;

	public int state;
	public static final int LOAD_DATABASE = 0;
	public static final int UPDATING_DATABASE = 1;
	public static final int CHECKING_AVAILABILITY = 2;
	public static final int FINISHING = 3;
	public static final int DONE = 4;

	public double updating_progress = 0;
	public double availability_progress = 0;

	public String updating_string = "";
	public String availability_string = "";

	public boolean auto_update = true;

	static {
		sources.add(new DataSource("EIDA_DE", "https://eida.bgr.de/fdsnws/station/1/query?nodata=404&", (byte) 0));
		sources.add(new DataSource("EIDA_RO", "https://eida-sc3.infp.ro/fdsnws/station/1/query?nodata=404&", (byte) 1));

		sources.add(new DataSource("RESIF", "https://ws.resif.fr/fdsnws/station/1/query?nodata=404&", (byte) 2));
		sources.add(
				new DataSource("GEOFON", "https://geofon.gfz-potsdam.de/fdsnws/station/1/query?nodata=404&", (byte) 3));
		sources.add(new DataSource("ERDE", "https://erde.geophysik.uni-muenchen.de/fdsnws/station/1/query?nodata=404&",
				(byte) 4));
		sources.add(new DataSource("ORFEUS", "https://www.orfeus-eu.org/fdsnws/station/1/query?nodata=404&", (byte) 5));
		sources.add(new DataSource("IRIS", "https://service.iris.edu/fdsnws/station/1/query?nodata=404&", (byte) 6));

		seedlinks.add(new SeedlinkNetwork((byte) 0, "Geofon Seedlink", GEOFON));
		seedlinks.add(new SeedlinkNetwork((byte) 1, "Resif Seedlink", RESIF));
		seedlinks.add(new SeedlinkNetwork((byte) 2, "Iris Seedlink", IRIS_RTSERVER));
	}

	private static File stationsFile = new File(Main.MAIN_FOLDER, "/stationDatabase/");

	private StationDatabase database;

	public StationManager() {

	}

	public void init() {
		load();
		update(false);
	}

	public int getState() {
		return state;
	}

	private void load() {
		state = LOAD_DATABASE;
		updating_progress = 0;
		updating_string = "Waiting...";
		availability_progress = 0;
		availability_string = "Waiting...";
		System.out.println("Loading station database...");
		File file = getDatabaseFile();
		if (!file.exists()) {
			confirmDialog("Info",
					"Station database file doesn't exists, a new one will be created at:\n" + file.getAbsolutePath(),
					JOptionPane.OK_CANCEL_OPTION, JOptionPane.INFORMATION_MESSAGE, "Cancel", "Ok");
			if (!getTempDatabaseFile().getParentFile().exists()) {
				getTempDatabaseFile().getParentFile().mkdirs();
			}
			updateDatabase();
		} else {
			try {
				ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
				StationDatabase database = (StationDatabase) in.readObject();
				in.close();
				this.database = database;
				System.out.println("Station database loaded sucesfully.");
				if (auto_update) {
					checkForUpdates();
				}
			} catch (Exception e) {
				confirmDialog("Error",
						exc(e) + "\nFailed to load database file " + file.getAbsolutePath()
								+ ", it is probably corrupted\nA new database will be created",
						JOptionPane.YES_NO_OPTION, JOptionPane.ERROR_MESSAGE, "Exit", "Continue");
				updateDatabase();
			}
		}
	}

	public void update(boolean updateDatabase) {
		if (updateDatabase) {
			updateDatabase();
		}

		for (SeedlinkNetwork seed : seedlinks) {
			seed.availableStations = 0;
			seed.selectedStations = 0;
		}

		checkAvailability();
		state = FINISHING;
		parseSelectedStations();

		int nets = 0;
		int stats = 0;
		int chans = 0;
		int ava = 0;
		int sel = 0;

		for (Network n : getDatabase().getNetworks()) {
			nets++;
			for (Station s : n.getStations()) {
				stats++;
				for (Channel ch : s.getChannels()) {
					chans++;
					if (ch.isAvailable()) {
						ava++;
					}
					if (ch.isSelected()) {
						sel++;
					}
				}
				Collections.sort(s.getChannels(), Comparator.comparing(Channel::getName));
			}
			Collections.sort(n.getStations(), Comparator.comparing(Station::getStationCode));
		}
		Collections.sort(getDatabase().getNetworks(), Comparator.comparing(Network::getNetworkCode));

		doImportantStuff();
		
	
		System.out.println();
		System.out.println(nets + " networks, " + stats + " stations, " + chans + " channels.");
		System.out.println(ava + " available channels, " + sel + " selected channels");
		System.out.println();
		System.out.println("The database is ready.");
		state = DONE;
		availability_string = "Done!";
	}
	
	private String exc(Exception e) {
		StringWriter sw = new StringWriter();
		PrintWriter pw = new PrintWriter(sw);
		e.printStackTrace(pw);
		String sStackTrace = sw.toString(); // stack trace as a string
		return sStackTrace;
	}

	public void save() {
		try {
			File file = getDatabaseFile();
			File _file = getTempDatabaseFile();
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(_file));
			System.out.println("Saving stations database to " + file.getAbsolutePath());
			out.writeObject(database);
			out.close();
			file.delete();
			_file.renameTo(file);
			_file.delete();
			System.out.println("Save successful.");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void doImportantStuff() {
		for (Network n : getDatabase().getNetworks()) {
			for (Station s : n.getStations()) {
				s.setNetwork(n);
				int i = 0;
				boolean b = false;
				for (Channel ch : s.getChannels()) {
					ch.setStation(s);
					if (ch.isSelected()) {
						if (!b) {
							s.setSelectedChannel(i);
							b = true;
						}
					}
					i++;
				}
				if (!b) {
					s.setSelectedChannel(-1);
				}
			}
		}
	}

	private void parseSelectedStations() {
		System.out.println("Parsing " + database.getSelectedStations().size() + " selected stations");
		for (SelectedStation selected : database.getSelectedStations()) {
			Channel ch = getChannel(selected.getNetworkCode(), selected.getStationCode(), selected.getChannelCode(),
					selected.getLocation());
			if (ch != null) {
				try {
					seedlinks.get(ch.getSeedlinkNetwork()).selectedStations++;
					ch.setSelected(true);
				} catch (IndexOutOfBoundsException e) {
					System.err.println("Selected channel " + selected.getNetworkCode() + " " + selected.getStationCode()
							+ " " + selected.getChannelCode() + " " + selected.getLocation()
							+ " is no longer available");
				}
			}
		}

	}

	private void checkAvailability() {
		state = CHECKING_AVAILABILITY;
		updating_progress = 1;
		updating_string = "Done!";
		for (Network n : getDatabase().getNetworks()) {
			for (Station s : n.getStations()) {
				for (Channel ch : s.getChannels()) {
					ch.setSeedlinkNetwork((byte) -1);
				}
			}
		}
		int i = 0;
		for (SeedlinkNetwork seed : seedlinks) {
			availability_progress = i / (double) seedlinks.size();
			System.out.println("Checking availability at " + seed.getHost() + "...");
			availability_string = "Checking availability at " + seed.getHost() + "...";
			try {
				SeedlinkReader reader = new SeedlinkReader(seed.getHost(), 18000, 90, false);
				availability(reader.getInfoString(SeedlinkReader.INFO_STREAMS), seed.getId());
				reader.close();
			} catch (Exception e) {
				confirmDialog("Error",
						exc(e) + "\nFailed to check availability at " + seed.getName() + " - " + seed.getHost(),
						JOptionPane.YES_NO_OPTION, JOptionPane.ERROR_MESSAGE, "Exit", "Ignore");
			}
			i++;
		}
		availability_progress = 1;
		availability_string = "Finishing...";
	}

	private void availability(String infoString, byte seedlinkNetwork) throws Exception {
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
				if (!isGoodChannel(channelName)) {
					continue;
				}
				String endDate = channel.getAttributes().getNamedItem("end_time").getTextContent();
				Calendar end = Calendar.getInstance();
				end.setTime(endDate.contains("-") ? format2.parse(endDate) : format3.parse(endDate));
				long delay = System.currentTimeMillis() - end.getTimeInMillis() - TimeFixer.offset();
				if (delay > 1000 * 60 * 60 * 24 * 7) {
					continue;
				}
				addAvailableChannel(networkCode, stationCode, channelName, locationCode, delay, seedlinkNetwork);
			}
		}

	}

	private void addAvailableChannel(String networkCode, String stationCode, String channelName, String locationCode,
			long delay, byte seedlinkNetwork) {
		Channel ch = getChannel(networkCode, stationCode, channelName, locationCode);
		if (ch != null && !ch.isAvailable()) {
			ch.setAvailable(true);
			ch.setDelay(delay);
			ch.setSeedlinkNetwork(seedlinkNetwork);

			seedlinks.get(seedlinkNetwork).availableStations++;
		} else {
			// available channel isnt in the database
		}
	}

	private Channel getChannel(String networkCode, String stationCode, String channelName, String locationCode) {
		Network net = database.getNetwork(networkCode);
		if (net != null) {
			Station stat = net.getStation(stationCode);
			if (stat != null) {
				return stat.getChannel(channelName, locationCode);
			} else {
				return null;
			}
		} else {
			return null;
		}
	}

	private void checkForUpdates() {
		System.out.println("Checking for database updates...");
		if (!needsUpdate()) {
			System.out.println("No need for database update.");
			return;
		} else {
			updateDatabase();
		}
	}

	public boolean needsUpdate() {
		return database == null || database.needsUpdate();
	}

	private void updateDatabase() {
		System.out.println("Updating database, this may take a minute...");

		Calendar now = Calendar.getInstance();
		now.setTime(new Date());

		StationDatabase oldDatabase = this.database;
		StationDatabase newDatabase = new StationDatabase(DATABASE_VERSION);
		this.database = newDatabase;
		state = UPDATING_DATABASE;
		updating_progress = 0;
		updating_string = "Waiting...";
		availability_progress = 0;
		availability_string = "Waiting...";

		int i = 0;
		for (DataSource se : sources) {
			updating_progress = i / (double) sources.size();
			try {
				downloadSource(se, newDatabase, now);
			} catch (Exception e) {
				confirmDialog("Error",
						exc(e) + "\nFailed to download stations from source " + se.getName() + "\n" + se.getUrl(),
						JOptionPane.YES_NO_OPTION, JOptionPane.ERROR_MESSAGE, "Exit", "Ignore");
			}
			i++;
		}
		newDatabase.logUpdate(now);
		if (oldDatabase != null) {
			newDatabase.copySelectedStationsFrom(oldDatabase);
		}
		save();

		updating_progress = 1;
		updating_string = "Done!";

		System.out.println("The new database now contains " + newDatabase.getNetworks().size() + " networks");
	}

	private void downloadSource(final DataSource se, StationDatabase database, Calendar now) throws Exception {
		URL url = new URL(se.getUrl()
				+ "level=channel&endafter="+format1.format(new Date())+"&includerestricted=false&format=xml&channel=" + channels);

		System.out.println("Connecting to " + se.getName());
		updating_string = "Connecting to " + se.getName();
		InputStream inp = url.openStream();

		DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
		f.setNamespaceAware(false);
		f.setValidating(false);
		final CountInputStream in = new CountInputStream(inp);

		in.setEvent(new Runnable() {
			public void run() {
				updating_string = "Downloading from " + se.getName() + ": " + (in.getCount() / 1024) + "kB";
			}
		});

		System.out.println("Downloading stations from " + se.getName() + " (" + url.toString() + ")");

		Thread counter = new Thread("Counting Thread") {
			@Override
			public void run() {
				while (true) {
					try {
						sleep(1000);
					} catch (InterruptedException e) {
						System.out.println("Total file size was " + (in.getCount() / 1024) + "kB");
						break;
					}
					System.out.println("Downloading... " + (in.getCount() / 1024) + "kB");
				}
			}
		};

		counter.start();

		Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);

		counter.interrupt();

		doc.getDocumentElement().normalize();

		Element root = doc.getDocumentElement();
		parseStations(se, database, root, now);
	}

	private void parseStations(DataSource se, StationDatabase database, Element root, Calendar now) {
		NodeList networks = root.getElementsByTagName("Network");
		for (int i = 0; i < networks.getLength(); i++) {
			try {
				String networkCode = obtainAttribute(networks.item(i), "code", "unknown");
				if (networkCode.equalsIgnoreCase("unknown")) {
					System.err.println("ERR: no network code wtf.");
					continue;
				}
				String networkDescription = obtainElement(networks.item(i), "Description", "");
				NodeList stations = ((Element) networks.item(i)).getElementsByTagName("Station");
				for (int j = 0; j < stations.getLength(); j++) {
					Node stationNode = stations.item(j);
					String stationCode = stationNode.getAttributes().getNamedItem("code").getNodeValue();
					String stationSite = ((Element) stationNode).getElementsByTagName("Site").item(0).getTextContent();
					NodeList channels = ((Element) stationNode).getElementsByTagName("Channel");
					for (int k = 0; k < channels.getLength(); k++) {
						try {

							/**
							 * Necessary values: lat lon alt sampleRate
							 * 
							 * Other can fail
							 */

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
							long sensitivity = -1;
							try {
								sensitivity = new BigDecimal(((Element) ((Element) (channelNode.getChildNodes()))
										.getElementsByTagName("InstrumentSensitivity").item(0))
												.getElementsByTagName("Value").item(0).getTextContent()).longValue();
							} catch (NullPointerException e) {
								System.err.println(
										"No Sensitivity!!!! " + stationCode + " " + networkCode + " " + channel);
							}
							double frequency = -1;
							try {
								frequency = Double.parseDouble(((Element) ((Element) (channelNode.getChildNodes()))
										.getElementsByTagName("InstrumentSensitivity").item(0))
												.getElementsByTagName("Frequency").item(0).getTextContent());
							} catch (NullPointerException e) {
								System.err.println("NO FREQUENCY!! " + stationCode + " " + networkCode + " " + channel);
							}
							double sampleRate = Double.parseDouble(((Element) channelNode)
									.getElementsByTagName("SampleRate").item(0).getTextContent());
							String inputUnits = null;
							try {
								inputUnits = ((Element) ((Element) ((Element) (channelNode.getChildNodes()))
										.getElementsByTagName("InstrumentSensitivity").item(0))
												.getElementsByTagName("InputUnits").item(0))
														.getElementsByTagName("Name").item(0).getTextContent();
							} catch (NullPointerException e) {
								System.err.println(
										"No Input Units!!! " + stationCode + " " + networkCode + " " + channel);
							}
							String startDate = "";
							try {
								startDate = obtainAttribute(channels.item(k), "startDate", "unknown");
							} catch (Exception e) {
								System.err.println("Error: broken start date.");
								continue;
							}

							String endDate = "";
							try {
								endDate = obtainAttribute(channels.item(k), "endDate", "unknown");
							} catch (Exception e) {

							}

							addChannel(database, se, networkCode, networkDescription, stationCode, stationSite, channel,
									locationCode, lat, lon, alt, sensitivity, frequency, sampleRate, inputUnits,
									startDate, endDate, now);
						} catch (Exception e) {
							System.err.println("Weird Error occured");
							e.printStackTrace();
							continue;
						}
					}
				}

			} catch (Exception e) {
				e.printStackTrace();
				continue;
			}
		}
	}

	private void addChannel(StationDatabase database, DataSource se, String networkCode, String networkDescription,
			String stationCode, String stationSite, String channel, String locationCode, double lat, double lon,
			double alt, long sensitivity, double frequency, double sampleRate, String inputUnits, String startDate,
			String endDate, Calendar now) {
		Calendar start = Calendar.getInstance();
		try {
			start.setTime(format1.parse(startDate));
		} catch (Exception e) {
			start = null;
		}

		// so its always null
		Calendar end = Calendar.getInstance();
		if (!endDate.equalsIgnoreCase("unknown")) {
			try {
				end.setTime(format1.parse(endDate));
				// it is temporary network
				if (end.get(Calendar.YEAR) < now.get(Calendar.YEAR) + 3) {
					return;
				}
			} catch (ParseException e) {
				e.printStackTrace();
			}
		}
		Network net = getOrCreateNetwork(database, networkCode, networkDescription);
		Station stat = net.getOrCreateStation(stationCode, stationSite, lat, lon, alt);

		long startMS = start == null ? -1 : start.getTimeInMillis();
		long endMS = end == null ? -1 : end.getTimeInMillis();

		if (!stat.containsChannel(channel, locationCode)) {
			Channel ch = new Channel(channel, locationCode, sensitivity, frequency, sampleRate, inputUnits, startMS,
					endMS, (byte) se.getId());
			stat.getChannels().add(ch);
		} else {
			System.err
					.println("Already Exists " + stationCode + " " + networkCode + " " + channel + ", " + locationCode);
		}

	}

	private Network getOrCreateNetwork(StationDatabase database2, String networkCode, String networkDescription) {
		for (Network net : database2.getNetworks()) {
			if (net.getNetworkCode().equals(networkCode)) {
				return net;
			}
		}
		Network net = new Network(networkCode, networkDescription);
		database2.getNetworks().add(net);
		return net;
	}

	private static boolean isGoodChannel(String channelName) {
		return (channelName.startsWith("E") || channelName.startsWith("S") || channelName.startsWith("H")
				|| channelName.startsWith("B")) && (channelName.charAt(1) == 'H');
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

	public StationDatabase getDatabase() {
		return database;
	}

	public static File getDatabaseFile() {
		return new File(stationsFile, "stationDatabase.dat");
	}

	public static File getTempDatabaseFile() {
		return new File(stationsFile, "_stationDatabase.dat");
	}

	@Override
	public void confirmDialog(String title, String message, int optionType, int messageType, String... options) {
		System.err.println(message);
	}

	public void editSelection(Station station, int selectedChannel) {
		if (selectedChannel == station.getSelectedChannel()) {
			return;// nothing changed
		} else {
			if (selectedChannel == -1) {
				for (Channel ch : station.getChannels()) {
					ch.setSelected(false);
				}
				seedlinks.get(station.getChannels().get(station.getSelectedChannel())
						.getSeedlinkNetwork()).selectedStations--;
				SelectedStation sel = getDatabase().getSelectedStation(station);
				if (sel == null) {
					System.err.println("Weird, no SelectedStation found.");
				} else {
					getDatabase().getSelectedStations().remove(sel);
				}
			} else {
				Channel theChannel = station.getChannels().get(selectedChannel);
				for (Channel ch : station.getChannels()) {
					ch.setSelected(ch.equals(theChannel));
				}
				seedlinks.get(theChannel.getSeedlinkNetwork()).selectedStations++;
				SelectedStation sel = getDatabase().getSelectedStation(station);
				if (sel == null) {
					// -1 => 5
					getDatabase().getSelectedStations().add(new SelectedStation(station.getNetwork().getNetworkCode(),
							station.getStationCode(), theChannel.getName(), theChannel.getLocationCode()));
				} else {
					sel.setChannelCode(theChannel.getName());
					sel.setLocation(theChannel.getLocationCode());
				}
			}
			station.setSelectedChannel(selectedChannel);
			save();
		}
	}

	public int getSelectedStations() {
		int n = 0;
		for (SeedlinkNetwork seed : seedlinks) {
			n += seed.selectedStations;
		}
		return n;
	}

}

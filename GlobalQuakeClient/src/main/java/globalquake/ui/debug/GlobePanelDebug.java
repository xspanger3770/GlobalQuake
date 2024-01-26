package globalquake.ui.debug;

import globalquake.core.GlobalQuake;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.quality.QualityClass;
import globalquake.main.Main;
import globalquake.ui.globalquake.feature.FeatureCities;
import globalquake.utils.GeoUtils;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import globalquake.ui.globalquake.EarthquakeListPanel;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.Point2D;
import globalquake.core.Settings;
import globalquake.utils.Scale;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.util.*;
import java.util.List;
import java.util.Timer;

public class GlobePanelDebug extends GQFrame {

	private final GlobePanel panel;
	private final JPanel list;
	private final JPanel mainPanel;
	private List<ArchivedQuake> archivedQuakes;
	protected boolean hideList;
	private boolean _containsListToggle;
	private boolean _containsSettings;

	public GlobePanelDebug() {
		createArchived();

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setPreferredSize(new Dimension(800, 600));

		panel = new GlobePanel(50,17) {

			@Override
			public void paint(Graphics gr) {
				super.paint(gr);
				Graphics2D g = (Graphics2D) gr;
				
				g.setColor(_containsListToggle ? Color.gray : Color.lightGray);
				g.fillRect(getWidth() - 20, 0, 20, 30);
				g.setColor(Color.black);
				g.drawRect(getWidth() - 20, 0, 20, 30);
				g.setFont(new Font("Calibri", Font.BOLD, 16));
				g.setColor(Color.black);
				g.drawString(hideList ? "<" : ">", getWidth() - 16, 20);
				
				g.setColor(_containsSettings ? Color.gray : Color.lightGray);
				g.fillRect(getWidth() - 20, getHeight() - 30, 20, 30);
				g.setColor(Color.black);
				g.drawRect(getWidth() - 20, getHeight() - 30, 20, 30);
				g.setFont(new Font("Calibri", Font.BOLD, 16));
				g.setColor(Color.black);
				g.drawString("S", getWidth() - 15, getHeight() - 8);

				String region = Regions.getRegion(getRenderer().getRenderProperties().centerLat, getRenderer().getRenderProperties().centerLon);
				g.setColor(Color.white);
				g.drawString(region, getWidth() / 2 - g.getFontMetrics().stringWidth(region), getHeight() - 16);

				double x = getWidth() / 2.0;
				double y = getHeight() / 2.0;
				double r = 10.0;
				g.draw(new Ellipse2D.Double(x - r/2, y - r/2, r, r));
			}
		};

		addKeyListener(new KeyAdapter() {
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_SPACE){
					panel.smoothTransition(0,0,0.5);
				}
				if(e.getKeyCode() == KeyEvent.VK_C){
					panel.setCinemaMode(!panel.isCinemaMode());
				}
			}
		});

		MonitorableCopyOnWriteArrayList<DebugStation> debugStations = new MonitorableCopyOnWriteArrayList<>();

		double centerLat = 50;
		double centerLon = 17;

		double maxDist = 10000;
		int total = 400;


		double phi = 1.61803398875;
		double c = maxDist / Math.sqrt(total);

		for(int n = 0; n < total; n++){
			double ang = 360.0 / (phi * phi) * n;
			double radius = Math.sqrt(n) * c;
			double[] latlon = GeoUtils.moveOnGlobe(centerLat, centerLon, radius, ang);
			debugStations.add(new DebugStation(new Point2D(latlon[0], latlon[1])));
		}

		System.out.println(debugStations.size());
		panel.getRenderer().addFeature(new FeatureDebugStation(debugStations));
		panel.getRenderer().addFeature(new FeatureCities());

		panel.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				if (x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30) {
					toggleList();
				}
			}
			
			@Override
			public void mouseExited(MouseEvent e) {
				_containsListToggle = false;
				_containsSettings = false;
			}
		});
		panel.addMouseMotionListener(new MouseAdapter() {

			@Override
			public void mouseMoved(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				_containsListToggle = x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30;
				_containsSettings = x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= panel.getHeight() - 30 && y <= panel.getHeight();
			}
			
		});

		list = new EarthquakeListPanel(this, archivedQuakes);
		panel.setPreferredSize(new Dimension(600, 600));
		list.setPreferredSize(new Dimension(300, 600));

		mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		mainPanel.setPreferredSize(new Dimension(800, 600));
		mainPanel.add(panel, BorderLayout.CENTER);
		mainPanel.add(list, BorderLayout.EAST);

		setContentPane(mainPanel);

		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setTitle("Globe Panel");

		Timer timer = new Timer();
		timer.scheduleAtFixedRate(new TimerTask() {
			public void run() {
				mainPanel.repaint();
			}
		}, 0, 1000 / 40);
	}

	private void createArchived() {
		archivedQuakes = new ArrayList<>();
		Random r = new Random();
		Settings.oldEventsTimeFilterEnabled = false;
		Settings.oldEventsMagnitudeFilterEnabled = false;
		for(double mag = 0.5; mag <= 11; mag += 0.2) {
			archivedQuakes.add(new ArchivedQuake(null, 0, 0, 0, mag, r.nextLong() % System.currentTimeMillis(), QualityClass.S, System.currentTimeMillis()+100)); //100ms added to make finalUpdateMillis > origin
		}
		//archivedQuakes.sort(Comparator.comparing(ArchivedQuake::getOrigin));
	}

	protected void toggleList() {
		hideList = !hideList;
		if (hideList) {
			panel.setSize(new Dimension(mainPanel.getWidth(), mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(0, (int) list.getPreferredSize().getHeight()));
		} else {
			panel.setSize(new Dimension(mainPanel.getWidth() - 300, mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(300, (int) list.getPreferredSize().getHeight()));
		}
		_containsListToggle = false;
		_containsSettings = false;
		revalidate();
	}

	public static void main(String[] args) {
		try {
			TauPTravelTimeCalculator.init();
			Regions.init();
			Scale.load();
			Sounds.load();
			GlobalQuake.prepare(Main.MAIN_FOLDER, null);
		} catch (Exception e) {
			return;
		}
		EventQueue.invokeLater(() -> new GlobePanelDebug().setVisible(true));
	}
}

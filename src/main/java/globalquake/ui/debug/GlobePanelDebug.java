package globalquake.ui.debug;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.globalquake.EarthquakeListPanel;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.Point2D;
import globalquake.utils.Scale;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.*;
import java.util.List;
import java.util.Timer;

public class GlobePanelDebug extends JFrame {

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

		panel = new GlobePanel() {

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
			}
		};
		Random r = new Random();
		MonitorableCopyOnWriteArrayList<DebugStation> debugStations = new MonitorableCopyOnWriteArrayList<>();
		for(int i = 0; i < 50; i++) {
			double x = 50 + r.nextDouble() * 10 - 5;
			double y = 17 + r.nextDouble() * 20 - 10;
			debugStations.add(new DebugStation(new Point2D(x, y)));
		}

		System.out.println(debugStations.size());
		panel.getRenderer().addFeature(new FeatureDebugStation(debugStations));

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

		list = new EarthquakeListPanel(archivedQuakes);
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
		for(double mag = 0.5; mag <= 11; mag += 0.5) {
			archivedQuakes.add(new ArchivedQuake(0, 0, 0, mag, r.nextLong() % System.currentTimeMillis()));
		}
		archivedQuakes.sort(Comparator.comparing(ArchivedQuake::getOrigin));
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
			Regions.init();
			Scale.load();
			Sounds.load();
		} catch (Exception e) {
			return;
		}
		EventQueue.invokeLater(() -> new GlobePanelDebug().setVisible(true));
	}
}

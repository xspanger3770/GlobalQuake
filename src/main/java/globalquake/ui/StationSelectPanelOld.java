package globalquake.ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.Locale;

import javax.swing.JOptionPane;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Station;

import globalquake.database.SeedlinkNetwork;
import globalquake.database.SeedlinkManager;
import globalquake.main.Main;

public class StationSelectPanelOld extends GlobePanelOld {

	private final StationSelect stationSelect;
	public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
	private static final SimpleDateFormat formatSimple = new SimpleDateFormat("yyyy/MM/dd");

	public StationSelectPanelOld(StationSelect stationSelect) {
		this.stationSelect = stationSelect;

		addMouseMotionListener(new MouseAdapter() {
			@Override
			public void mouseMoved(MouseEvent e) {
				StationSelectPanelOld.super.repaint();
			}
		});

		addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				if (lastMouse != null) {
					ArrayList<Station> clickedStations = new ArrayList<>();
					for (Station s : stationSelect.getDisplayedStations()) {
						double x = getX(s.getLat(), s.getLon());
						double y = getY(s.getLat(), s.getLon());
						if (isMouseNearby(x, y, 7)) {
							clickedStations.add(s);
						}
					}
					if (!clickedStations.isEmpty()) {
						editStations(clickedStations);
						StationSelectPanelOld.super.repaint();
					}
				}
			}
		});
	}

	protected void editStations(ArrayList<Station> clickedStations) {
		Station station = null;

		if (clickedStations.size() == 1) {
			station = clickedStations.get(0);
		} else {
			String[] realOptions = new String[clickedStations.size()];
			int i = 0;
			for (Station s : clickedStations) {
				realOptions[i] = s.getStationCode() + " " + s.getNetwork().getNetworkCode();
				i++;
			}

			String result = (String) JOptionPane.showInputDialog(this, "Select station to edit:", "Station selection",
					JOptionPane.PLAIN_MESSAGE, null, realOptions, realOptions[0]);
			if (result == null) {
				return;
			} else {
				int i2 = 0;
				for (String s : realOptions) {
					if (s.equals(result)) {
						station = clickedStations.get(i2);
						break;
					}
					i2++;
				}
			}
		}

		if (station == null) {
			System.err.println("Fatal Error: null");
        } else {
			int ava = 0;
			for (Channel ch : station.getChannels()) {
				if (ch.isAvailable()) {
					ava++;
				}
			}

			int k = 0;
			int ki = 0;
			String[] availableChannels = new String[ava + 1];
			availableChannels[0] = "None";
			int selectedIndex = 0;
			HashMap<Integer, Integer> map = new HashMap<>();
			for (Channel ch : station.getChannels()) {
				if (ch.isAvailable()) {
					Calendar start = Calendar.getInstance();
					start.setTimeInMillis(ch.getStart());
					String str = ch.getLocationCode() + " " + ch.getName() + " - " + ch.getSampleRate() + "sps, "
							+ (ch.getSource() < 0 ? "?????, "
									: SeedlinkManager.sources.get(ch.getSource()).getName() + ", ")
							+ (ch.getSeedlinkNetwork() == -1 ? "not available"
									: SeedlinkManager.seedlinks.get(ch.getSeedlinkNetwork()).getHost())
							+ ", begin: " + formatSimple.format(start.getTime()) + ", delay="
							+ f1d.format(ch.getDelay() / 1000.0) + "s";

					availableChannels[k + 1] = str;
					if (ki == station.getSelectedChannel()) {
						selectedIndex = k + 1;
					}
					map.put(k + 1, ki);
					k++;
				}
				ki++;
			}

			String result = (String) JOptionPane.showInputDialog(this,
					"Select channel for station " + station.getStationCode() + ":", "Channel selection",
					JOptionPane.PLAIN_MESSAGE, null, availableChannels, availableChannels[selectedIndex]);
            if (result != null) {
				int selectedChannel = -1;
				int l = 0;
				for (String str : availableChannels) {
					if (str.equals(result)) {
						selectedChannel = l;
						break;
					}
					l++;
				}
				// if "None" is selected, then it is -1
				Station s2 = station;
				int see = selectedChannel;
				new Thread(() -> {
					try {
						stationSelect.getStationManager().editSelection(s2, see == -1 || see == 0 ? -1 : map.get(see));
					} catch (IOException e) {
						Main.getErrorHandler().handleException(e);
					}
				}).start();
			}
        }

	}

	@Override
	public void paint(Graphics gr) {
		super.paint(gr);
		Graphics2D g = (Graphics2D) gr;
		for (Station s : stationSelect.getDisplayedStations()) {
			double x = getX(s.getLat(), s.getLon());
			double y = getY(s.getLat(), s.getLon());
			if (!isOnScreen(x, y)) {
				continue;
			}
			double r = 12;
			Polygon pol = new Polygon();
			g.setColor(s.isSelected() ? Color.green : Color.lightGray);
			boolean mouse = isMouseNearby(x, y, 7);
			if (mouse) {
				g.setColor(Color.white);
			}
			pol.addPoint((int) (x - r / 2), (int) (y + r / 3));
			pol.addPoint((int) (x + r / 2), (int) (y + r / 3));
			pol.addPoint((int) (x), (int) (y - r / 2));
			g.fill(pol);

			if (mouse) {
				g.setColor(Color.white);
				String str = s.getLat() + ", " + s.getLon() + ", " + s.getAlt() + "m - " + s.getStationSite();

				g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y - r * 0.5 - 3 - 0));

				str = s.getStationCode() + " [" + s.getNetwork().getNetworkCode() + "] - "
						+ s.getNetwork().getDescription();

				g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y - r * 0.5 - 3 - 12));

				int __y = 26;
				int i = 0;
				for (Channel ch : s.getChannels()) {
					g.setColor(!ch.isAvailable() ? Color.LIGHT_GRAY
							: s.getSelectedChannel() == i ? Color.green : Color.cyan);
					Calendar start = Calendar.getInstance();
					start.setTimeInMillis(ch.getStart());
					str = ch.getLocationCode() + " " + ch.getName() + " - " + ch.getSampleRate() + "sps, "
							+ (ch.getSource() < 0 ? "?????, "
									: SeedlinkManager.sources.get(ch.getSource()).getName() + ", ")
							+ (ch.getSeedlinkNetwork() == -1 ? "not available"
									: SeedlinkManager.seedlinks.get(ch.getSeedlinkNetwork()).getHost())
							+ ", begin: " + formatSimple.format(start.getTime()) + ", delay="
							+ f1d.format(ch.getDelay() / 1000.0) + "s";
					g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5),
							(int) (y - r * 0.5 - 3 + __y));
					y += 12;
					i++;
				}
			}
		}

		int _y = 16;
		for (SeedlinkNetwork seed : SeedlinkManager.seedlinks) {
			g.setColor(seed.selectedStations > 0 ? Color.green : Color.lightGray);
			g.setFont(new Font("Calibri", Font.BOLD, 18));
			g.drawString(seed.getHost() + " (" + seed.selectedStations + "/" + seed.availableStations + ")", 5, _y);
			_y += 20;
		}

	}

}

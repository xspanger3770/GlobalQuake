package globalquake.ui;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;
import globalquake.database_old.SeedlinkManager;
import globalquake.main.Main;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Timer;
import java.util.TimerTask;

public abstract class DatabaseMonitorFrameOld extends JFrame {

	private final SeedlinkManager seedlinkManager;
	private final JLabel lblVersion;
	private final JLabel lblLastUpdate;
	private final JProgressBar progressBar2;
	private final JProgressBar progressBar1;
	private final JButton btnLaunch;
	private final JButton btnSelectStations;
	private final JLabel lblNetworks;
	private final JLabel lblStations;
	private final JLabel lblChannels;
	private final JLabel lblSelected;
	private final JLabel lblAvailable;
	private final DefaultTreeModel defaultTreeModel;
	private final JButton btnUpdate;
	private final JTree tree;
	private StationSelectOld stationSelectOld;

	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

	public DatabaseMonitorFrameOld(SeedlinkManager seedlinkManager) {
		this.seedlinkManager = seedlinkManager;

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JPanel contentPane = new JPanel();
		contentPane.setPreferredSize(new Dimension(460, 330));
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		btnLaunch = new JButton("Launch " + Main.fullName);
		btnLaunch.setEnabled(false);
		btnLaunch.setBounds(230, 285, 218, 32);
		btnLaunch.setMargin(new Insets(0, 0, 0, 0));
		contentPane.add(btnLaunch);

		btnLaunch.addActionListener(e -> {
            if (seedlinkManager.getSelectedStations() == 0) {
                String[] options = { "Cancel", "Yes" };
                int n = JOptionPane.showOptionDialog(DatabaseMonitorFrameOld.this, "Launch with no selected stations?",
                        "Confirm", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, options,
                        options[0]);
                if (n == 0) {
                    return;
                }
            }
            DatabaseMonitorFrameOld.this.dispose();
            DatabaseMonitorFrameOld.this.launch();
        });

		lblVersion = new JLabel("Database Version: Loading...");
		lblVersion.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblVersion.setBounds(10, 25, 238, 20);
		contentPane.add(lblVersion);

		lblLastUpdate = new JLabel("Last Update: Loading...");
		lblLastUpdate.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblLastUpdate.setBounds(10, 5, 238, 20);
		contentPane.add(lblLastUpdate);

		progressBar1 = new JProgressBar();
		progressBar1.setIndeterminate(true);
		progressBar1.setString("0% - Waiting...");
		progressBar1.setStringPainted(true);
		progressBar1.setValue(0);
		progressBar1.setBounds(10, 200, 438, 32);
		contentPane.add(progressBar1);

		progressBar2 = new JProgressBar();
		progressBar2.setIndeterminate(true);
		progressBar2.setString("0% - Waiting...");
		progressBar2.setStringPainted(true);
		progressBar2.setBounds(10, 240, 438, 32);
		contentPane.add(progressBar2);

		btnSelectStations = new JButton("Select Stations");
		btnSelectStations.setEnabled(false);
		btnSelectStations.setBounds(10, 285, 210, 32);
		contentPane.add(btnSelectStations);

		btnSelectStations.addActionListener(e -> EventQueue.invokeLater(() -> {
            if (seedlinkManager == null) {
                return;
            }
            DatabaseMonitorFrameOld.this.setEnabled(false);
            stationSelectOld = new StationSelectOld(seedlinkManager);
            stationSelectOld.setVisible(true);
            stationSelectOld.addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosing(WindowEvent e1) {
                    DatabaseMonitorFrameOld.this.setEnabled(true);
                }
            });
        }));

		lblNetworks = new JLabel("Networks: Loading...");
		lblNetworks.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblNetworks.setBounds(10, 45, 238, 20);
		contentPane.add(lblNetworks);

		lblStations = new JLabel("Stations: Loading...");
		lblStations.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblStations.setBounds(10, 65, 210, 20);
		contentPane.add(lblStations);

		lblChannels = new JLabel("Channels: Loading...");
		lblChannels.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblChannels.setBounds(10, 85, 210, 20);
		contentPane.add(lblChannels);

		lblSelected = new JLabel("Selected stations: Loading...");
		lblSelected.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblSelected.setBounds(10, 125, 210, 20);
		contentPane.add(lblSelected);

		lblAvailable = new JLabel("Available stations: Loading...");
		lblAvailable.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblAvailable.setBounds(10, 105, 210, 20);
		contentPane.add(lblAvailable);

		DefaultMutableTreeNode wait = new DefaultMutableTreeNode("Loading...");

		defaultTreeModel = new DefaultTreeModel(wait);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(258, 5, 190, 183);
		contentPane.add(scrollPane);
		tree = new JTree(defaultTreeModel);
		tree.setEnabled(false);
		scrollPane.setViewportView(tree);

		btnUpdate = new JButton("Update");
		btnUpdate.setBounds(10, 155, 238, 32);
		btnUpdate.setEnabled(false);
		contentPane.add(btnUpdate);

		btnUpdate.addActionListener(e -> {
            if (seedlinkManager != null && seedlinkManager.getDatabase() != null) {
                new Thread(() -> {
					try {
						seedlinkManager.updateDatabase();
					} catch (IOException ex) {
						Main.getErrorHandler().handleException(ex);
					}
				}).start();
            }
        });

		pack();
		setTitle("Station Database");
		setResizable(false);
		setLocationRelativeTo(null);
		runTimer();
	}

	private void runTimer() {
		final Timer timer;
		timer = new Timer();

		TimerTask task = new TimerTask() {

			@Override
			public void run() {
				updateFrame();
			}
		};

		timer.schedule(task, 50, 50);

		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				timer.cancel();
			}
		});
	}

	public void confirmDialog(String title, String message, int optionType, int messageType, String... options) {
		int n = JOptionPane.showOptionDialog(this, message, title, optionType, messageType, null, options, options[0]);
		if (n == 0 && options.length > 1) {
			System.exit(0);
		}
	}

	private void updateFrame() {
		int state = seedlinkManager.getState();
        if (seedlinkManager.getDatabase() == null) {
            return;
        }
        if (state < SeedlinkManager.DONE && btnSelectStations.isEnabled()) {
            btnSelectStations.setEnabled(false);
            btnLaunch.setEnabled(false);
            btnUpdate.setEnabled(false);
            tree.setEnabled(false);
        }
        if (state <= 0) {
            lblVersion.setText("Database version: Loading...");
            lblNetworks.setText("Networks: Loading...");
            lblStations.setText("Stations: Loading...");
            lblChannels.setText("Channels: Loading...");
            lblSelected.setText("Selected: Loading...");
            lblLastUpdate.setText("Last update: Loading...");
            lblAvailable.setText("Available: Loading...");
        }

        progressBar2.setIndeterminate(state <= 0);
        progressBar1.setIndeterminate(state <= 0);

        if (state > 0) {
            if (seedlinkManager.getDatabase().getDatabaseVersion() == SeedlinkManager.DATABASE_VERSION) {
                lblVersion.setText("<html>Database version: <font color='green'>"
                        + seedlinkManager.getDatabase().getDatabaseVersion() + "</font></html>");
            } else {
                lblVersion.setText("<html>Database version: <font color='red'>"
                        + seedlinkManager.getDatabase().getDatabaseVersion() + " -> "
                        + SeedlinkManager.DATABASE_VERSION + "</font></html>");
            }
            Calendar c = Calendar.getInstance();
            c.setTimeInMillis(seedlinkManager.getDatabase().getLastUpdate());
            lblLastUpdate.setText("<html>Last update: "
                    + (state == SeedlinkManager.UPDATING_DATABASE ? "<font color='green'>Updating now...</font>"
                            : "<font color='" + colAge(c) + "'>" + dateFormat.format(c.getTime()) + "</font>")
                    + "</html>");
            progressBar1.setValue((int) (seedlinkManager.updating_progress * 100));
            progressBar1.setString(
                    (int) (seedlinkManager.updating_progress * 100) + "% - " + seedlinkManager.updating_string);

            progressBar2.setValue((int) (seedlinkManager.availability_progress * 100));
            progressBar2.setString((int) (seedlinkManager.availability_progress * 100) + "% - "
                    + seedlinkManager.availability_string);

            int nets = 0;
            int stats = 0;
            int channels = 0;
            int ava = 0;
            int sel = 0;

			seedlinkManager.getDatabase().getNetworksReadLock().lock();

			try {
				for (Network n : seedlinkManager.getDatabase().getNetworks()) {
					nets++;
					for (Station s : n.getStations()) {
						stats++;
						boolean av = false;
						boolean se = false;
						for (Channel ch : s.getChannels()) {
							channels++;
							if (ch.isAvailable() && !av) {
								ava++;
								av = true;
							}
							if (ch.isSelected() && !se) {
								sel++;
								se = true;
							}
						}
					}
				}
			} finally {
				seedlinkManager.getDatabase().getNetworksReadLock().unlock();
			}

            lblNetworks.setText("Networks: " + nets);
            lblStations.setText("Stations: " + stats);
            lblChannels.setText("Channels: " + channels);
            lblAvailable.setText("Available stations: " + ava);
            if (state >= SeedlinkManager.DONE) {
                lblSelected.setText("Selected stations: " + sel);
            }
            if (state >= SeedlinkManager.DONE && !btnSelectStations.isEnabled()) {
                btnSelectStations.setEnabled(true);
                btnLaunch.setEnabled(true);
                btnUpdate.setEnabled(true);
                tree.setEnabled(true);
                defaultTreeModel.setRoot(createRoot());
            }
        }
        // revalidate();
		repaint();
	}

	private String colAge(Calendar c) {
		double ageDays = (System.currentTimeMillis() - c.getTimeInMillis()) / (1000.0 * 60 * 60 * 24);
		if (ageDays >= 7) {
			return "red";
		}
		if (ageDays >= 3) {
			return "orange";
		}
		return "green";
	}

	private DefaultMutableTreeNode createRoot() {
		DefaultMutableTreeNode networksNode = new DefaultMutableTreeNode("Networks");
		seedlinkManager.getDatabase().getNetworksReadLock().lock();
		try {
			for (Network n : seedlinkManager.getDatabase().getNetworks()) {
				DefaultMutableTreeNode networkNode = new DefaultMutableTreeNode(n.getNetworkCode());
				for (Station s : n.getStations()) {
					DefaultMutableTreeNode statNode = new DefaultMutableTreeNode(s.getStationCode());
					for (Channel ch : s.getChannels()) {
						DefaultMutableTreeNode chanNode = new DefaultMutableTreeNode(
								ch.getName() + " " + ch.getLocationCode());
						statNode.add(chanNode);
					}
					networkNode.add(statNode);
				}

				networksNode.add(networkNode);
			}
		} finally {
			seedlinkManager.getDatabase().getNetworksReadLock().unlock();
		}

		return networksNode;
	}

	public abstract void launch();
}

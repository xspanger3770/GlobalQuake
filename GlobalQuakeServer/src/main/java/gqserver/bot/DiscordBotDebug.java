package gqserver.bot;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import gqserver.main.Main;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.entities.Guild;
import net.dv8tion.jda.api.entities.Role;
import net.dv8tion.jda.api.entities.channel.concrete.TextChannel;
import net.dv8tion.jda.api.events.session.ReadyEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.requests.GatewayIntent;
import org.jetbrains.annotations.NotNull;

@SuppressWarnings("all")
public class DiscordBotDebug extends ListenerAdapter {

    private static JDA jda;

    public static void main(String[] args) {
        GlobalQuake.prepare(Main.MAIN_FOLDER, null);
        jda = JDABuilder.createDefault(Settings.discordBotToken)
                .enableIntents(GatewayIntent.GUILD_MESSAGES)
                .addEventListeners(new DiscordBotDebug())
                .build();
    }

    @Override
    public void onReady(@NotNull ReadyEvent event) {
        for (Guild guild : jda.getGuilds()) {
            System.out.printf("GUILD %s - %s\n".formatted(guild.getName(), guild.getId()));
            for (Role role : guild.getRoles()) {
                System.out.printf("     ROLE %s - %s\n".formatted(role.getName(), role.getId()));
            }
            for (TextChannel channel : guild.getTextChannels()) {
                System.out.printf("     CHANNEL %s - %s\n".formatted(channel.getName(), channel.getId()));
            }
        }

        /*Guild guild = jda.getGuildById("1146784713806188614");
        TextChannel textChannel = guild.getTextChannelsByName("general", true).get(0);

        textChannel.sendMessage("M15 Incoming").queue();*/

       /* Guild guild = jda.getGuildById("955128942078087228");

        for(var member:guild.getRoles()){
            System.err.println(member.getName()+", "+member.getId());
        }

        List<Role> roles = new ArrayList<>();

        roles.addAll(guild.getRolesByName("Ping 4.0+", false));

        StringBuilder stringBuilder = new StringBuilder();

        for(Role role : roles){
            stringBuilder.append(role.getAsMention());
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("PLS WORK");
        builder.setDescription("randomrandomrandom");


        //jda.getGuildById("955128942078087228").getTextChannelById("955128942078087230").sendMessage(stringBuilder.toString()).queue();
        jda.getGuildById("955128942078087228").getTextChannelById("955128942078087230")
                .sendMessageEmbeds(builder.build()).queue();*/

        // jda.getGuildById("1150807505115549736").getTextChannelById("1151878913123962940").createInvite().queue(invite -> System.err.println(invite.getUrl()));
    }
}
